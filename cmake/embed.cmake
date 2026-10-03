# Turns the shared simulation source and the GPU kernels into C strings compiled into brtrain,
# so the program is a single self-contained executable.
function(embed var file)
  file(READ "${file}" hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
  file(APPEND "${OUT}" "static const char ${var}_data[] = {${hex}0x00};\nconst char* ${var} = ${var}_data;\n")
endfunction()
file(WRITE "${OUT}" "/* generated from src/sim_core.h and src/kernels — do not edit */\n")
embed(BR_SRC_SIM_CORE "${SRC}/sim_core.h")
embed(BR_SRC_METAL "${SRC}/kernels/metal_kernel.metal")
embed(BR_SRC_OPENCL "${SRC}/kernels/opencl_kernel.cl")
embed(BR_UI_HTML "${SRC}/../web/index.html")
