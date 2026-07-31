# ---------------------------
# linux terminal command
# ---------------------------
#  apt install python3-torch
#  cmake --build . --config Release
#  rm -r ./build/ || true && cmake -S . -B ./build/
#  rm -r ./build/ && cmake -S . -B ./build/
#  cmake -S . -B ./build/
#  cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
#  cp readMarket.cpp ./build/ && cmake --build ./build --config Release && ./build/readMarket
#  cp readMarket.cpp ./build/ && cp apple-inc-appl.csv ./build/ && cmake --build ./build --config Release && ./build/readMarket
#  cp readMarket.cpp ./build/ && cmake --build ./build --config Release && ./build/readMarket > output.log && nano output.log
#
#  g++ readMarket.Generated.cpp -o readMarket.Generated
#  ./readMarket.Generated.cpp
#
# ---------------------------

