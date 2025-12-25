# Lichess chessboard screenshots conversion to FEN using a Convolutional Neural Network
Developing a CNN to recognize lichess pieces from a screenshot and create the related FEN, since many puzzle/opening/training videos on youtube do not include it nor the PGN.  
The goal is to build a model strong enough to recognize all the pieces from various pieces set in several backgrounds, in order to evaluate a position without having to recreate it manually using the board editor.

## Jupyter Notebook info
All the zipped files need to be unzipped to be used correctly as in the code.  
Feel free to copy the CNN architecture and state_dict (CNN_bleeding) and try it yourself on your screenshots after adding them inside the Screenshot_Test folder.  
Keep in mind that the screenshot borders need to be as accurate as possible to the chessboard boarders, or else weird cropping might happen.  
The CNN has been trained using GPU so it might raise an error when only CPU is available. In the "best model loading" inside Jupiter notebook i added the code to load the tensors on CPU even though sometimes it still raises errors on Colab.  

The CNN has been trained avoiding the following pieces sets:
```
anarcandy      too weirdly shaped
disguised      too weirdly shaped
horsey         too weirdly shaped
kiwen-suwi     too weirdly shaped, knights are the other way around
letter         too weirdly shaped
mono           monochromatic
reillycraig    too small
shapes         too weirdly shaped
```
All the material as been gathered following this link: https://github.com/lichess-org/lila/tree/master/public/piece  
Since a few months has passed by, there might be some piece sets not included in the folder i give, feel free to add them in the Lichess_Pieces_Sets folder and do your evaluation on whether keeping them or not.   

The whole Jupyter Notebook is gathering the data connecting to Google Drive, except for the DataLoader part in which i copy the data directly into Colab to massively speed up the training phase.  
As long as your Drive looks like this:
```
.
└── content/
    └── gdrive/
        └── MyDrive/
            └── Projects/
                └── Lichess_Screenshot_To_FEN/
                    ├── Lichess_Piece_Sets/
                    │   └── (folders with pieces sets)
                    ├── Screenshot_Test/
                    │   └── (chessboard screenshots)
                    └── fen_to_image/
                        ├── white/
                        │   └── (white pieces)
                        └── black/
                            └── (black pieces)
```
there should not be any problem running the code.  
The final result once the code has fully run should look like this:
```
.
└── content/
    └── gdrive/
        └── MyDrive/
            └── Projects/
                └── Lichess_Screenshot_To_FEN/
                    ├── dataset/
                    │   ├── test/
                    │   │   └── (test folder for each piece)
                    │   └── train/
                    │       └── (train folder for each piece)
                    ├── Lichess_Piece_Sets/
                    │   └── (folders with pieces sets)
                    ├── Grouped_Pieces/
                    │   └── (folder for each piece)
                    ├── Screenshot_Test/
                    │   ├── (chessboard screenshots)
                    │   └── Resized_Test/
                    │       └── (resized screenshots)
                    └── fen_to_image/
                        ├── white/
                        │   └── (white pieces)
                        └── black/
                            └── (black pieces)
```
Keep in mind that if you want to test the network on your screenshots, you need to comply to the following notation for the images in the Screenshot_Test folder:
- 00.*extension*
- 01.*extension*
- ...
- 10.*extension*  
(if more than 100 then 000.*extension*, i am sure you got the gist of it).  

If you want to check the performances on the screenshot dataset, at the end of the project i used the following metrics:
- per-board confusion matrix
- per-board average accuracy
- ~~per-board per-class accuracy~~
- per-board per-class precision
- per-board per-class recall
- per-board per-class f1 score
- ~~overall per-class average accuracy~~
- overall accuracy
- overall per-class average recall
- overall per-class average precision
- overall f1 score  
 
This is my first semi-serious approach to data cleaning and neural networks, with respect to CNN and images.  
I know i overlooked various aspects but for the programming knowledge i hold right now, and the time i can dedicate to personal projects, i consider what i have done more than enough to grow in that sense.  

Below you can find the metrics computed from the predictions on *61* chessboard screenshots:
```
Overall average accuracy = 0.981, with 3830/3904 total correct predictions

Overall average piece precision (given everything i said to be of class x, how many were correct)
Empty piece precision = 0.975
Black Pawn piece precision = 0.984
White Pawn piece precision = 1.0
Black Bishop piece precision = 0.836
White Bishop piece precision = 0.902
Black Knight piece precision = 0.836
White Knight piece precision = 0.623
Black Rook piece precision = 0.951
White Rook piece precision = 0.929
Black Queen piece precision = 0.893
White Queen piece precision = 0.878
Black King piece precision = 0.951
White King piece precision = 1.0

Overall average piece recall (given everything of class x, how many i got right)
Empty piece recall = 0.998
Black Pawn piece recall = 0.884
White Pawn piece recall = 0.957
Black Bishop piece recall = 0.828
White Bishop piece recall = 0.902
Black Knight piece recall = 0.836
White Knight piece recall = 0.623
Black Rook piece recall = 0.885
White Rook piece recall = 0.926
Black Queen piece recall = 0.902
White Queen piece recall = 0.918
Black King piece recall = 0.951
White King piece recall = 1.0

Overall average piece F1-Score (Measures how well the model performs, in this context better than using accuracy)
Empty piece F1-Score = 0.986
Black Pawn piece F1-Score = 0.918
White Pawn piece F1-Score = 0.974
Black Bishop piece F1-Score = 0.831
White Bishop piece F1-Score = 0.902
Black Knight piece F1-Score = 0.836
White Knight piece F1-Score = 0.623
Black Rook piece F1-Score = 0.907
White Rook piece F1-Score = 0.926
Black Queen piece F1-Score = 0.896
White Queen piece F1-Score = 0.889
Black King piece F1-Score = 0.951
White King piece F1-Score = 1.0
Overall average F1-Score = 0.895
```
These metrics might be slightly inflated due to the fact that several screenshots are the same but with a different background and piece set. This decision was made to meet a time constraint, since assigning for each square its real class for 61 completely different boards would have been too time consuming.

PS.  
**Sometimes the import section requires a restart due to conflicts with different Pillow dependencies. Do as suggested by Colab since i am not aware of a solution from such problem.  
Another problem might arises when loading the .pth since the network is trained on GPU but then, in an unknown way, does not let you load it back into GPU for testing. Just use CPU for that part since you don't need anything else but the imports, CNN definition and target_names prior to that part of the project.  
The CNN .pth is called bleeding because i add bleeding to the images, creating in the training set images blank spaces in one or two randomly selected side in order to better classify misaligned pieces derived from poorly screenshot cropping.**

## Python fen.py
I also added the .py code to run the CNN model directly on your screenshots, one at a time.  
Keep in mind that the screenshot size does not matter since i apply resize to make them (800,800), as long as they tend to capture just the chessboard as good as they can.  
In the following portion i left the commands you need to run to make the project workable on your station:  

Access via terminal this folder and create the venv (virtual enviroment) inside of it
```
python -m venv venv
```
Activate the venv
```
venv\Scripts\activate
```
Install the required libraries
```
pip install torch torchvision numpy opencv-python matplotlib pillow cairosvg fentoboardimage
```
The commands above need to be executed only the first time you create the venv. Every other time you will access the folder via terminal you can simply run:  

Activate the venv (only one time per terminal)
```
venv\Scripts\activate
```
Run the script
```
python fen.py
```

Once running, you will simply need to follow these steps:
- Drag and drop your screenshot into the terminal
- An image will display the way the network sees each cell
- Close the image to get the FEN
- An image will display the reconstructed chessboard from the predicted FEN, to help you better visualize possible errors.
- Close the image to terminate the execution.
