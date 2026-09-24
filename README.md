# Kyra Riedel - RISO Print Color Approximation Tool
My project is based around the idea of Riso-graphic printing, which applies the inks in layers instead of combining the cartridges for each pixel. Currently software exists to process the images into specific color files (https://colorshift.theretherenow.com/profiles), but does not determine the best match and limits the user to the existing color profiles.
My idea is that given a photo, number of colors, and available inks, the program would be able to find the top-3 best/worst ranked RISO approximations displayed side-by-side and it would display how they were created along with the accuracy score of each. It would also produce the individual layers and new image to the user for download. I would not need to use any data sets (choose from my own photos to show the best range of the photos) or any additional computing resources. 

Build (from the repository root, which contains `CMakeLists.txt` and the `.cpp` / `.h` sources):
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

Executable: `./riso_tool` (inside `build/` by default). The Haar cascade is copied next to the binary. Separation files are written to `./riso_output` relative to the current working directory.

## Run

### Interactive mode

```bash
./riso_tool
```

### CLI args mode

```bash
./riso_tool --image /path/to/image.jpg --layers 3
./riso_tool --image /path/to/image.jpg --layers 3 --inks 0,3,5
./riso_tool --image /path/to/image.jpg --layers 3 --inks all
./riso_tool --image /path/to/image.jpg --layers 3 --inks 0,3,5 --edit
```

Links:
https://drive.google.com/file/d/1IfUk-WfXRNl3w3VAUPZMJCLoDaei6sCt/view?usp=sharing
Note: not said elsewhere I believe, but the photo all over the report is Rizo from Survivor, it was a pun, but since I didn't explain it in my video it just looks a little creepy.
