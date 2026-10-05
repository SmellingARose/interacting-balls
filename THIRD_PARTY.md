# Third-party software

- **three.js** r160 (`web/vendor/`): MIT License, Copyright 2010-2023 three.js authors. The license text is in
  `web/vendor/three-LICENSE.txt`. The app serves these files itself, so the interface needs no internet connection.
  Only `three.module.js` and the OrbitControls / Line2 addons the 3D view uses are included.
- Fonts: the interface and the explainer pages name Saira Condensed, JetBrains Mono and IBM Plex Sans but do not load or
  include them; without them installed the pages fall back to system fonts.
- `src/br_cl.h` declares the few OpenCL functions the program calls (loaded from the GPU driver at run time); it
  contains no Khronos header text.
