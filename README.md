# Lichess_Screenshot_To_FEN
 Developing a CNN to recognize lichess pieces from a screenshot and create the related FEN, since many puzzle/opening/training videos on youtube do not include it nor the PNG. 
 
The goal is to build a model strong enough to recognize all the pieces from various pieces set and pieces backgrounds, in order to evaluate a position without having to recreate it manually using the board editor.

All the zipped files need to be unzipped to be used correctly as in the code. 
Feel free to copy the CNN architecture and state_dict (CNN.pth) and try it yourself on your screenshots after addig them inside the Screenshot_Test folder. 
Keep in mind that the screenshot borders need to be as accurate as possible to the chessboard boarders, or else weird cropping might happen.
The CNN has been trained using GPU so it might raise an error when only CPU is available. In the "best model loading" inside Jupiter notebook i added the code to load the tensors on CPU.

The CNN has been trained avoiding the following pieces sets:
anarcandy      too weirdly shaped
disguised      too weirdly shaped
horsey         too weirdly shaped
kiwen-suwi     too weirdly shaped, knights are the other way around
letter         too weirdly shaped
mono           monochromatic
reillycraig    too small
shapes         too weirdly shaped

The whole Jupiter notebook is gathering the data by connecting to my Google Drive, except for the DataLoader part in which i copy the data directly into Colab to massively speed up the training part.
As long as your Drive looks like this:
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
there shouldn't be any problem running the code.
The final result once the code is fully run should look like this:
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
