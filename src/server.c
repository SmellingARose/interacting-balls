// Local UI server: serves the trainer's interface and a small JSON API on 127.0.0.1 only, then opens the interface
// in its own app window (Edge on Windows, Chrome on macOS if installed, otherwise the default browser).
#include "threads.h"   // must come first on Windows (winsock2 before windows.h)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trainer.h"
#ifdef _WIN32
  #include <ws2tcpip.h>
  #include <shellapi.h>
  typedef SOCKET sock_t;
  #define CLOSESOCK closesocket
#else
  #include <sys/socket.h>
  #include <sys/select.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int sock_t;
  #define CLOSESOCK close
  #define INVALID_SOCKET (-1)
#endif

extern const char* BR_UI_HTML;
static volatile int quitReq = 0;
static double lastRequest = 0;
int trainer_busy(void);

static int send_all(sock_t s, const char* p, size_t n) {
  while (n) { int k = (int)send(s, p, (int)(n > 1 << 20 ? 1 << 20 : n), 0); if (k <= 0) return -1; p += k; n -= (size_t)k; }
  return 0;
}
static void reply(sock_t s, int code, const char* type, const char* body, size_t len) {
  char h[256];
  int n = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %lu\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                   code, code == 200 ? "OK" : code == 404 ? "Not Found" : "Error", type, (unsigned long)len);
  send_all(s, h, (size_t)n); if (len) send_all(s, body, len);
}
static void reply_sb(sock_t s, Sb* b) { reply(s, 200, "application/json", b->s ? b->s : "{}", b->s ? b->n : 2); free(b->s); }

static void handle(sock_t s) {
  // read headers (+ body by Content-Length)
  size_t cap = 65536, n = 0; char* buf = (char*)malloc(cap + 1); char* body = NULL; size_t clen = 0;
  for (;;) {
    int k = (int)recv(s, buf + n, (int)(cap - n), 0); if (k <= 0) { free(buf); return; } n += (size_t)k; buf[n] = 0;
    char* e = strstr(buf, "\r\n\r\n");
    if (e) {
      char* cl = strstr(buf, "Content-Length:"); if (!cl) cl = strstr(buf, "content-length:");
      clen = cl ? (size_t)strtoul(cl + 15, NULL, 10) : 0;
      size_t hdr = (size_t)(e + 4 - buf);
      if (clen > 64u << 20) { reply(s, 413, "text/plain", "too large", 9); free(buf); return; }
      if (hdr + clen > cap) { cap = hdr + clen; buf = (char*)realloc(buf, cap + 1); }
      while (n < hdr + clen) { k = (int)recv(s, buf + n, (int)(cap - n), 0); if (k <= 0) break; n += (size_t)k; }
      buf[n] = 0; body = buf + hdr; break;
    }
    if (n == cap) { cap *= 2; buf = (char*)realloc(buf, cap + 1); }
  }
  // only answer requests addressed to this machine (blocks DNS-rebinding tricks from web pages)
  char* host = strstr(buf, "\r\nHost:"); if (!host) host = strstr(buf, "\r\nhost:");
  if (!host || !(strstr(host, "127.0.0.1") == host + 8 || strstr(host, "localhost") == host + 8)) { reply(s, 403, "text/plain", "forbidden", 9); free(buf); return; }
  char method[8] = "", path[256] = "";
  sscanf(buf, "%7s %255s", method, path);
  lastRequest = br_now();
  char* q = strchr(path, '?'); char query[128] = ""; if (q) { snprintf(query, sizeof query, "%s", q + 1); *q = 0; }
  Sb out = {0}; TrainCfg c;
  if (!strcmp(path, "/") || !strcmp(path, "/index.html")) reply(s, 200, "text/html; charset=utf-8", BR_UI_HTML, strlen(BR_UI_HTML));
  else if (!strcmp(path, "/favicon.ico")) reply(s, 204, "image/x-icon", "", 0);
  else if (!strcmp(path, "/api/hardware")) { hardware_json(&out); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/status")) { int since = 0; sscanf(query, "since=%d", &since); trainer_status_json(&out, since); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/start")) { cfg_defaults(&c); cfg_from_json(&c, body); trainer_start(&c); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/pause")) { trainer_pause(); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/reset")) { trainer_reset(); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/star")) { trainer_star_json(&out); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/load")) { char m[200]; int r = trainer_load_json(body ? body : "", m, sizeof m);
    sb_printf(&out, "{\"ok\":%s,\"message\":", r ? "false" : "true"); sb_jstr(&out, m); sb_printf(&out, "}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/autotune")) { cfg_defaults(&c); cfg_from_json(&c, body); autotune_start(&c); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/autotune/status")) { autotune_status_json(&out); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/mode")) {   // the mode switch: save this mode's star, load the other's (training applies it itself)
    cfg_defaults(&c); cfg_from_json(&c, body ? body : ""); if (!trainer_busy()) trainer_set_mode(c.mode);
    sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/fly")) { fly_json(body ? body : "", &out); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/battle")) { battle_json(body ? body : "", &out); reply_sb(s, &out); }
  else if (!strcmp(path, "/api/quit")) { sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); quitReq = 1; }
  else reply(s, 404, "text/plain", "not found", 9);
  free(buf);
}

static void open_window(const char* url) {
  char cmd[600];
#if defined(_WIN32)
  // Edge ships with Windows; --app gives a standalone window without tabs or address bar
  snprintf(cmd, sizeof cmd, "--app=%s", url);
  if ((INT_PTR)ShellExecuteA(NULL, "open", "msedge", cmd, NULL, SW_SHOWNORMAL) <= 32) ShellExecuteA(NULL, "open", url, NULL, NULL, SW_SHOWNORMAL);
#elif defined(__APPLE__)
  FILE* f = fopen("/Applications/Google Chrome.app/Contents/Info.plist", "r");
  if (f) { fclose(f); snprintf(cmd, sizeof cmd, "open -na 'Google Chrome' --args --app='%s' >/dev/null 2>&1", url); }
  else snprintf(cmd, sizeof cmd, "open '%s'", url);
  if (system(cmd) != 0) { snprintf(cmd, sizeof cmd, "open '%s'", url); system(cmd); }
#else
  snprintf(cmd, sizeof cmd, "xdg-open '%s' >/dev/null 2>&1 &", url); if (system(cmd)) {}
#endif
}

int serve_ui(int openWindow) {
#ifdef _WIN32
  WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
#endif
  sock_t ls = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
  struct sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int port = 0;
  // Already running? Then just open its window and leave (one trainer per computer).
  { sock_t t = socket(AF_INET, SOCK_STREAM, 0); a.sin_port = htons(8642);
    if (connect(t, (struct sockaddr*)&a, sizeof a) == 0) {
      const char* rq = "GET /favicon.ico HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"; char r[64] = "";
      send(t, rq, (int)strlen(rq), 0); recv(t, r, sizeof r - 1, 0); CLOSESOCK(t);
      if (strstr(r, " 204 ")) { if (openWindow) open_window("http://127.0.0.1:8642/"); printf("Already running: opened its window.\n"); return 0; }
    } else CLOSESOCK(t); }
  for (int p = 8642; p < 8662; p++) { a.sin_port = htons((unsigned short)p); if (bind(ls, (struct sockaddr*)&a, sizeof a) == 0) { port = p; break; } }
  if (!port || listen(ls, 16) != 0) { fprintf(stderr, "Could not open a local port for the interface.\n"); return 1; }
  char url[64]; snprintf(url, sizeof url, "http://127.0.0.1:%d/", port);
  printf("Ball Arena trainer is running.\nInterface: %s\nClose this window or press Quit in the interface to stop.\n", url); fflush(stdout);
  trainer_init();
  if (openWindow) open_window(url);
  lastRequest = br_now();
  while (!quitReq) {
    // wake every second: quit by itself once idle (no window open, nothing training) for 2 minutes
    fd_set fs; FD_ZERO(&fs); FD_SET(ls, &fs); struct timeval tv = { 1, 0 };
    if (select((int)ls + 1, &fs, NULL, NULL, &tv) <= 0) { if (!trainer_busy() && br_now() - lastRequest > 120) break; continue; }
    sock_t s = accept(ls, NULL, NULL); if (s == INVALID_SOCKET) continue;
    handle(s); CLOSESOCK(s);
  }
  trainer_pause();
  CLOSESOCK(ls);
  return 0;
}
