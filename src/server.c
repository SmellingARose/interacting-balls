// Local UI server: serves the trainer's interface and a small JSON API on 127.0.0.1 only, then opens the interface
// in its own app window (Edge on Windows, Chrome on macOS if installed, otherwise the default browser).
#include "threads.h"   // must come first on Windows (winsock2 before windows.h)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
  #include <sys/time.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int sock_t;
  #define CLOSESOCK close
  #define INVALID_SOCKET (-1)
#endif

extern const char* BR_UI_HTML;
// three.js and the few addons the page uses, embedded in the program (the page loads nothing from the internet)
extern const char *BR_V_THREE, *BR_V_ORBIT, *BR_V_LINE2, *BR_V_LINEMAT, *BR_V_LINEGEO, *BR_V_SEG2, *BR_V_SEGGEO;
static const struct { const char* path; const char** data; } VENDOR[] = {
  { "/vendor/three.module.js", &BR_V_THREE }, { "/vendor/addons/controls/OrbitControls.js", &BR_V_ORBIT },
  { "/vendor/addons/lines/Line2.js", &BR_V_LINE2 }, { "/vendor/addons/lines/LineMaterial.js", &BR_V_LINEMAT },
  { "/vendor/addons/lines/LineGeometry.js", &BR_V_LINEGEO }, { "/vendor/addons/lines/LineSegments2.js", &BR_V_SEG2 },
  { "/vendor/addons/lines/LineSegmentsGeometry.js", &BR_V_SEGGEO } };
// Every route and the one method it accepts. Commands change state, so they need POST, which brings the X-BR check.
enum { R_PAGE, R_ICON, R_VENDOR, R_STATUS, R_HARDWARE, R_STAR, R_TUNESTATUS,
       R_START, R_PAUSE, R_RESET, R_LOAD, R_MODE, R_TUNE, R_FLY, R_BATTLE, R_QUIT };
static const struct { const char* path; int id, cmd; } ROUTES[] = {
  { "/", R_PAGE, 0 }, { "/index.html", R_PAGE, 0 }, { "/favicon.ico", R_ICON, 0 }, { "/api/status", R_STATUS, 0 },
  { "/api/hardware", R_HARDWARE, 0 }, { "/api/star", R_STAR, 0 }, { "/api/autotune/status", R_TUNESTATUS, 0 },
  { "/api/start", R_START, 1 }, { "/api/pause", R_PAUSE, 1 }, { "/api/reset", R_RESET, 1 }, { "/api/load", R_LOAD, 1 },
  { "/api/mode", R_MODE, 1 }, { "/api/autotune", R_TUNE, 1 }, { "/api/fly", R_FLY, 1 }, { "/api/battle", R_BATTLE, 1 },
  { "/api/quit", R_QUIT, 1 } };
#define HDR_MAX (64u << 10)     // request line and headers: browsers send every cookie for 127.0.0.1, whatever the port
#define BODY_MAX (64u << 20)    // a saved network
#define CONN_MAX 32             // connections served at once
static volatile int quitReq = 0;
static int g_port = 0;
static br_mutex cmdMx;          // commands run one at a time, as before (starting, pausing and resetting are not reentrant)
static br_mutex liveMx;         // guards the two below
static int live = 0; static double lastRequest = 0;
int trainer_busy(void);

static void set_timeout(sock_t s, int sec) {
#ifdef _WIN32
  DWORD ms = (DWORD)sec * 1000; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof ms);
#else
  struct timeval tv = { sec, 0 }; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}
static int send_all(sock_t s, const char* p, size_t n) {
  while (n) { int k = (int)send(s, p, (int)(n > 1 << 20 ? 1 << 20 : n), 0); if (k <= 0) return -1; p += k; n -= (size_t)k; }
  return 0;
}
static const char* reason(int code) {
  switch (code) { case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request"; case 403: return "Forbidden";
    case 404: return "Not Found"; case 405: return "Method Not Allowed"; case 408: return "Request Timeout"; case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large"; case 503: return "Service Unavailable"; default: return "Error"; }
}
// `extra`: further header lines, each ending in \r\n
static void reply_h(sock_t s, int code, const char* type, const char* extra, const char* body, size_t len) {
  char h[1024];
  int n = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %lu\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n%sConnection: close\r\n\r\n",
                   code, reason(code), type, (unsigned long)len, extra ? extra : "");
  send_all(s, h, (size_t)n); if (len) send_all(s, body, len);
}
static void reply(sock_t s, int code, const char* type, const char* body, size_t len) { reply_h(s, code, type, NULL, body, len); }
static void reply_text(sock_t s, int code, const char* msg) { reply(s, code, "text/plain", msg, strlen(msg)); }
static void reply_sb(sock_t s, Sb* b) { reply(s, 200, "application/json", b->s ? b->s : "{}", b->s ? b->n : 2); free(b->s); }
// The page may load only its own scripts and talk only to this server, and no other page may frame it.
static const char* PAGE_HEADERS =
  "Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
  "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\nX-Frame-Options: DENY\r\n";

// Value of header `name` (case-insensitive) copied into out; returns 0 when missing. Looks only at the header lines.
static int header_value(const char* req, const char* name, char* out, int len) {
  size_t nl = strlen(name); const char* p = req;
  while ((p = strstr(p, "\r\n")) != NULL) {
    p += 2; if (p[0] == '\r') break;   // blank line: end of headers
    int match = 1; for (size_t i = 0; i < nl; i++) { char a = p[i], b = name[i]; if (a >= 'A' && a <= 'Z') a += 32; if (b >= 'A' && b <= 'Z') b += 32; if (a != b) { match = 0; break; } }
    if (!match || p[nl] != ':') continue;
    p += nl + 1; while (*p == ' ' || *p == '\t') p++;
    int n = 0; while (p[n] && p[n] != '\r' && p[n] != '\n' && n < len - 1) { out[n] = p[n]; n++; } out[n] = 0;
    while (n > 0 && (out[n - 1] == ' ' || out[n - 1] == '\t')) out[--n] = 0;
    return 1;
  }
  return 0;
}
// "127.0.0.1", "localhost" or either with :port.
static int loopback_host(const char* h) {
  const char* names[2] = { "127.0.0.1", "localhost" };
  for (int i = 0; i < 2; i++) { size_t n = strlen(names[i]);
    if (!strncmp(h, names[i], n)) { const char* r = h + n; if (!*r) return 1; if (*r == ':' && r[1]) { for (r++; *r; r++) if (*r < '0' || *r > '9') return 0; return 1; } } }
  return 0;
}
// This page's own origin: http://127.0.0.1:<port> or http://localhost:<port>, the port this server is bound to.
static int own_origin(const char* o) {
  char a[64], b[64]; snprintf(a, sizeof a, "http://127.0.0.1:%d", g_port); snprintf(b, sizeof b, "http://localhost:%d", g_port);
  return !strcmp(o, a) || !strcmp(o, b);
}
// Only answer requests addressed to this machine, from this app's own page:
//  - the Host header must be 127.0.0.1 or localhost (optionally with a port): blocks DNS rebinding;
//  - any Origin header must be this page's exact origin (scheme, name and port): other pages, other local servers included,
//    are refused;
//  - commands must be POSTs that carry X-BR: 1. A custom header makes a browser ask permission first (a CORS preflight),
//    which this server never grants, and a link, image or form cannot add one: blocks other web pages from sending commands.
static int header_ok(const char* req, int cmd) {
  char v[300];
  if (!header_value(req, "Host", v, sizeof v) || !loopback_host(v)) return 0;
  if (header_value(req, "Origin", v, sizeof v) && !own_origin(v)) return 0;
  if (cmd && (!header_value(req, "X-BR", v, sizeof v) || strcmp(v, "1"))) return 0;
  if (cmd && header_value(req, "Sec-Fetch-Site", v, sizeof v) && !strcmp(v, "cross-site")) return 0;
  return 1;
}

// Reads one request into a NUL-terminated buffer: headers up to HDR_MAX within 10 s, then the body by Content-Length within
// 60 s (each recv waits at most 3 s, so a client sending a byte at a time cannot hold the connection). Replies to bad requests
// itself and returns NULL.
static char* read_request(sock_t s, char** body) {
  double t0 = br_now(); size_t cap = HDR_MAX, n = 0, hdr = 0, clen = 0; char v[32];
  char* buf = (char*)malloc(cap + 1); if (!buf) return NULL;
  for (;;) {
    if (n == cap) { reply_text(s, 431, "headers too large"); free(buf); return NULL; }
    if (br_now() - t0 > 10) { reply_text(s, 408, "timeout"); free(buf); return NULL; }
    int k = (int)recv(s, buf + n, (int)(cap - n), 0); if (k <= 0) { free(buf); return NULL; }
    size_t from = n > 3 ? n - 3 : 0;   // only the new bytes, and the end mark may straddle two reads
    n += (size_t)k; buf[n] = 0;
    char* e = strstr(buf + from, "\r\n\r\n"); if (e) { hdr = (size_t)(e + 4 - buf); break; }
  }
  if (header_value(buf, "Transfer-Encoding", v, sizeof v)) { reply_text(s, 400, "chunked bodies are not supported"); free(buf); return NULL; }
  if (header_value(buf, "Content-Length", v, sizeof v)) {
    char* end; unsigned long long l = strtoull(v, &end, 10);
    if (!v[0] || *end || v[0] == '-' || v[0] == '+') { reply_text(s, 400, "bad length"); free(buf); return NULL; }
    if (l > BODY_MAX) { reply_text(s, 413, "too large"); free(buf); return NULL; }
    clen = (size_t)l; }
  if (hdr + clen > cap) { char* nb = (char*)realloc(buf, hdr + clen + 1); if (!nb) { reply_text(s, 413, "too large"); free(buf); return NULL; } buf = nb; cap = hdr + clen; }
  while (n < hdr + clen) {   // a body cut short is dropped, not half-applied
    if (br_now() - t0 > 60) { reply_text(s, 408, "timeout"); free(buf); return NULL; }
    int k = (int)recv(s, buf + n, (int)(hdr + clen - n), 0); if (k <= 0) { free(buf); return NULL; } n += (size_t)k; }
  buf[hdr + clen] = 0; *body = buf + hdr;
  return buf;
}

static void handle(sock_t s) {
  char* body = NULL; char* buf = read_request(s, &body); if (!buf) return;
  char method[8] = "", path[256] = "";
  sscanf(buf, "%7s %255s", method, path);
  char* q = strchr(path, '?'); char query[128] = ""; if (q) { snprintf(query, sizeof query, "%s", q + 1); *q = 0; }
  int id = -1, cmd = 0;
  if (!strncmp(path, "/vendor/", 8)) id = R_VENDOR;
  else for (size_t i = 0; i < sizeof ROUTES / sizeof ROUTES[0]; i++) if (!strcmp(path, ROUTES[i].path)) { id = ROUTES[i].id; cmd = ROUTES[i].cmd; break; }
  char v[300];
  if (!header_value(buf, "Host", v, sizeof v) || !loopback_host(v)) { reply_text(s, 403, "forbidden"); free(buf); return; }
  if (id < 0) { reply_text(s, 404, "not found"); free(buf); return; }
  if (strcmp(method, cmd ? "POST" : "GET")) {   // OPTIONS too: a preflight is never granted
    reply_h(s, 405, "text/plain", cmd ? "Allow: POST\r\n" : "Allow: GET\r\n", "method not allowed", 18); free(buf); return; }
  if (!header_ok(buf, cmd)) { reply_text(s, 403, "forbidden"); free(buf); return; }
  br_lock(&liveMx); lastRequest = br_now(); br_unlock(&liveMx);
  Sb out = {0}; TrainCfg c;
  // reads that only take the trainer's lock run alongside a command, so a long Pause or Reset does not freeze the page
  int serial = cmd || id == R_HARDWARE;
  if (serial) br_lock(&cmdMx);
  switch (id) {
    case R_PAGE: reply_h(s, 200, "text/html; charset=utf-8", PAGE_HEADERS, BR_UI_HTML, strlen(BR_UI_HTML)); break;
    case R_ICON: reply(s, 204, "image/x-icon", "", 0); break;
    case R_VENDOR: { int hit = 0;
      for (size_t i = 0; i < sizeof VENDOR / sizeof VENDOR[0]; i++) if (!strcmp(path, VENDOR[i].path)) { reply(s, 200, "text/javascript; charset=utf-8", *VENDOR[i].data, strlen(*VENDOR[i].data)); hit = 1; break; }
      if (!hit) { reply_text(s, 404, "not found"); }
      break; }
    case R_HARDWARE: hardware_json(&out); reply_sb(s, &out); break;
    case R_STATUS: { int since = 0; sscanf(query, "since=%d", &since); trainer_status_json(&out, since); reply_sb(s, &out); break; }
    case R_STAR: trainer_star_json(&out); reply_sb(s, &out); break;
    case R_TUNESTATUS: autotune_status_json(&out); reply_sb(s, &out); break;
    case R_START: cfg_defaults(&c); cfg_from_json(&c, body); trainer_start(&c); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); break;
    case R_PAUSE: trainer_pause(); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); break;
    case R_RESET: cfg_defaults(&c); cfg_from_json(&c, body); trainer_reset(&c); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); break;   // the page's settings name the mode and matchup
    case R_LOAD: { char m[200] = ""; int r = trainer_load_json(body, m, sizeof m);
      sb_printf(&out, "{\"ok\":%s,\"message\":", r ? "false" : "true"); sb_jstr(&out, m); sb_printf(&out, "}"); reply_sb(s, &out); break; }
    case R_TUNE: cfg_defaults(&c); cfg_from_json(&c, body); autotune_start(&c); sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); break;
    case R_MODE:   // the mode switch: save this mode's star, load the other's (training applies it itself)
      cfg_defaults(&c); cfg_from_json(&c, body); if (!trainer_busy()) trainer_set_mode(c.mode);
      sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); break;
    case R_FLY: fly_json(body, &out); reply_sb(s, &out); break;
    case R_BATTLE: battle_json(body, &out); reply_sb(s, &out); break;
    case R_QUIT: sb_printf(&out, "{\"ok\":true}"); reply_sb(s, &out); quitReq = 1; break;
  }
  if (serial) br_unlock(&cmdMx);
  free(buf);
}

// One short-lived thread per connection: a slow or idle connection (browsers open spare ones) cannot hold up the others.
static void serve_conn(sock_t s) {
  set_timeout(s, 3); handle(s); CLOSESOCK(s);
  br_lock(&liveMx); live--; br_unlock(&liveMx);
}
#ifdef _WIN32
static DWORD WINAPI conn_thread(LPVOID p) { serve_conn((sock_t)(uintptr_t)p); return 0; }
static int spawn_conn(sock_t s) {
  HANDLE h = CreateThread(NULL, 4u << 20, conn_thread, (LPVOID)(uintptr_t)s, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
  if (!h) return -1;
  CloseHandle(h); return 0;
}
#else
static void* conn_thread(void* p) { serve_conn((sock_t)(intptr_t)p); return NULL; }
static int spawn_conn(sock_t s) {   // detached, with a roomy stack (handlers used to run on the main thread)
  pthread_t t; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED); pthread_attr_setstacksize(&at, 4u << 20);
  int r = pthread_create(&t, &at, conn_thread, (void*)(intptr_t)s); pthread_attr_destroy(&at); return r ? -1 : 0;
}
#endif

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
#ifdef _WIN32
  // Windows' SO_REUSEADDR would let this bind a port another program already listens on; exclusive use makes it move on
  int yes = 1; setsockopt(ls, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&yes, sizeof yes);
#else
  int yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
#endif
  struct sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int port = 0;
  // Already running? Then just open its window and leave (one trainer per computer).
  { sock_t t = socket(AF_INET, SOCK_STREAM, 0); a.sin_port = htons(8642);
    if (connect(t, (struct sockaddr*)&a, sizeof a) == 0) {
      const char* rq = "GET /favicon.ico HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"; char r[64] = "";
      set_timeout(t, 2);   // another program on this port may never answer
      send(t, rq, (int)strlen(rq), 0); recv(t, r, sizeof r - 1, 0); CLOSESOCK(t);
      if (strstr(r, " 204 ")) { if (openWindow) open_window("http://127.0.0.1:8642/"); printf("Already running: opened its window.\n"); return 0; }
    } else CLOSESOCK(t); }
  for (int p = 8642; p < 8662; p++) { a.sin_port = htons((unsigned short)p); if (bind(ls, (struct sockaddr*)&a, sizeof a) == 0) { port = p; break; } }
  if (!port || listen(ls, 16) != 0) { fprintf(stderr, "Could not open a local port for the interface.\n"); return 1; }
  g_port = port; br_mutex_init(&cmdMx); br_mutex_init(&liveMx);
  char url[64]; snprintf(url, sizeof url, "http://127.0.0.1:%d/", port);
  printf("Ball Arena trainer is running.\nInterface: %s\nClose this window or press Quit in the interface to stop.\n", url); fflush(stdout);
  trainer_init();
  if (openWindow) open_window(url);
  lastRequest = br_now();
  while (!quitReq) {
    // wake every second: quit by itself once idle (no window open, nothing training) for 2 minutes
    fd_set fs; FD_ZERO(&fs); FD_SET(ls, &fs); struct timeval tv = { 1, 0 };
    if (select((int)ls + 1, &fs, NULL, NULL, &tv) <= 0) {
      br_lock(&liveMx); int idle = !live && br_now() - lastRequest > 120; br_unlock(&liveMx);
      if (idle && !trainer_busy()) break;
      continue; }
    sock_t s = accept(ls, NULL, NULL); if (s == INVALID_SOCKET) continue;
    br_lock(&liveMx); int full = live >= CONN_MAX; if (!full) live++; br_unlock(&liveMx);
    if (full) { set_timeout(s, 1); reply_text(s, 503, "busy"); CLOSESOCK(s); continue; }
    if (spawn_conn(s)) serve_conn(s);   // no thread: serve it here, as before
  }
  CLOSESOCK(ls);
  br_lock(&cmdMx); trainer_pause(); br_unlock(&cmdMx);   // after any command still running
  for (int i = 0; i < 50; i++) { br_lock(&liveMx); int n = live; br_unlock(&liveMx); if (!n) break; br_sleep_ms(100); }   // let replies finish
  return 0;
}
