# CudaGalaxies
A CUDA powered n-body simulation to model Disk-Bulge-Halo galaxies based on the [Kiujiken and Dubinski paper](https://arxiv.org/abs/astro-ph/9502051)

Initial conditions for the Disk, Bulge, and Halo are generated with the same author's ["GalactICS" package](http://adsabs.harvard.edu/abs/2011ascl.soft09011K)
  
Where a few changes to generating files and removal of certain calls as specified in the above link were made, 
particularly in order to to get an easy number of points to perform operations on in CUDA; multiples of 32, 
specifically 16384 and 32768 points. 

The code is an implimentation of Leapfrog integration, slightly different from the Verlet method used by 
NVIDIA in their n-body sample code which comes with most CUDA installations. Other significant changes 
I have made for this simplified model include:
- No openGL interoperability is utilized; instead data is transfered to the host and saved to file for later animation
- Did away with templates for variables data inputs, floating precision is good for now. 
- Changed the integration algorithm as mentioned above. 
- Forced cooldown times have been added for larger simulations, which are doing their best to burn my laptop ;) 

NVIDIA's write-up can be found in their ["gems" text](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch31.html)

### Making and Running 
I've provided what a physics B.S. might call a Makefile, with nearly all non-standard libraries being part of the CUDA 8.0 package which is available [through NVIDIA](https://developer.nvidia.com/cuda-downloads)

When running there is one input argument:
./main **input**
Where **input.data** is a 1 line header, 7 column text file like the ones generated by the GalactICS package
and/or the one's I have provided.

To properly run a simulation, change the running parameters in main as desired:
- **numPoints**: The number of points in the initial data file you are going to run from 
- **steps**: Steps of the simulation
- **saveStep**: Of **steps** you want to save every **saveStep** to file, main determinant of runtime
- **coolStep** and **sleepTime**: How many steps to manually pause after, and how long to pause for
this one was for me personally, running all of this on my Lenovo laptop which gets quite hot quite fast.
- **softSquared**: For an N-Body simulation to be realistic it needs a softening parameter for it's interactions
to limit the effects of close encounters

For each saveStep *position* data is then saved to **input*Run*.data** for later animation, which I will discuss elsewhere.