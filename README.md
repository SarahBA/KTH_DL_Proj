# Code structure

The two main source files are as follows :    
- The code for the style transfer algorithm can be found in `src/styleTransfer.py`.      
- `ScipyOptimizer.py` simply acts as an interface between Keras/Tensorflow and scipy. It is a dependency used by `styleTransfer.py`.

Moreover :
- The file `batchTransfer.py` allows us to batch run the style transfer algorithm.  
- `denoise.py` performs total variation denoising on an imput image.   


# Running the code

First, make sure that Tensorflow and Keras are properly installed. Install all the requirements in the `requirements.txt` file with pip.  

In order to perform style transfer on a singleimage, simply run the following in the terminal : `python3 styleTransfer.py -s <pathToStyleImage> -c <pathToContentImage>`.    
Some other options are also available, type `python3 styleTransfer.py --help` for details.   

If you wish to automate the style transfer process and schedule several runs of the algorithm, modify and run `batchTransfer.py` accordingly. Details are provided in the surce file.

Some sample images used in the report have been provided in the `images` folder.
