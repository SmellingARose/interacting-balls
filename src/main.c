// Ballistic Range trainer. Started without options (double-click), it opens the interface; all training,
// auto-tuning and flights run natively on this computer. Developer options (--bench, --compare, --train) exist
// for testing.
#include <stdio.h>
#include <string.h>
#include "trainer.h"
int serve_ui(int openWindow);

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], "--bench") || !strcmp(argv[i], "--compare") || !strcmp(argv[i], "--train") || !strcmp(argv[i], "--list-devices") || !strcmp(argv[i], "--help"))
      return cli_main(argc, argv);
  int open = 1; for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--no-window")) open = 0;
  return serve_ui(open);
}
