# Lichess screenshots conversion to FEN using a Convolutional Neural Network
Developing a CNN to recognize lichess pieces from a screenshot and create the related FEN, since many puzzle/opening/training videos on youtube do not include it nor the PGN. 
 
The goal is to build a model strong enough to recognize all the pieces from various pieces set in several backgrounds, in order to evaluate a position without having to recreate it manually using the board editor.

All the zipped files need to be unzipped to be used correctly as in the code.  
Feel free to copy the CNN architecture and state_dict (CNN.pth and CNN_bleeding) and try it yourself on your screenshots after adding them inside the Screenshot_Test folder.  
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

The whole Jupiter notebook is gathering the data connecting to Google Drive, except for the DataLoader part in which i copy the data directly into Colab to massively speed up the training phase.  
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
- per-board per-class accuracy
- per-board per-class precision
- per-board per-class recall
- overall per-class average accuracy
- overall per-class average recall
- overall per-class average precision
 
This is my first semi-serious approach to data cleaning and neural networks, with respect to CNN and images.  
I know i overlooked various aspects but for the programming knowledge i hold right now, and the time i can dedicate to personal projects, i consider what i have done more than enough to grow in that sense.

PS.  
**Sometimes the import section requires a restart due to conflicts with different Pillow dependencies. Do as suggested by Colab since i am not aware of a solution from such problem.  
Another problem might arises when loading the .pth since the network is trained on GPU but then, in an unknown way, does not let you load it back inot GPU. Just use CPU for that part since you don't need anything else but the imports, CNN definition and target_names prior to that part of the project.**
