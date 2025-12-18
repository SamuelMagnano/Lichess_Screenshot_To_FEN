import os
import cv2
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from fentoboardimage import fenToImage, loadPiecesFolder

#CNN definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        test_tensor = torch.zeros(1, 3, 100, 100)
        with torch.no_grad():
            conv_out = self.conv(test_tensor)

        flatten_dim = conv_out.numel()

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.out(x)
        return x

#Target classes
target_names = [
    "wP","bP",
    "wN","bN",
    "wB","bB",
    "wR","bR",
    "wQ","bQ",
    "wK","bK","em"
]

device = torch.device("cpu")


#Inout transform definition (just normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])


#Screenshot decomposition into 64 cells
def cells_from_screenshot(image_path):
    image = cv2.imread(image_path)

    square_size = 100
    parts = []

    for i in range(0, 800, square_size):
        for j in range(0, 800, square_size):
            square = image[i:i + square_size, j:j + square_size]
            parts.append(square)

    image_path = image_path.split("/")[-1]
    print(f"Image: {image_path}")

    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=.09)

    for ax, img in zip(grid, parts):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    plt.show()
    return parts


#FEN classical notation
def fen_notation(piece):
    match piece:
        case "wP": return "P"
        case "bP": return "p"
        case "wN": return "N"
        case "bN": return "n"
        case "wB": return "B"
        case "bB": return "b"
        case "wR": return "R"
        case "bR": return "r"
        case "wQ": return "Q"
        case "bQ": return "q"
        case "wK": return "K"
        case "bK": return "k"
        case "em": return "E"


#FEN from complete CNN predictions
def fen_from_predictions(predictions, additional_info):
  pieces = np.array(predictions[0]).reshape(8,8)
  confidence = np.array(predictions[1]).reshape(8,8)
  fen = ""
  space = " "
  fixed_conclusion = "- 0 1"
  threshold = .5
  chessboard = []
  #print(f"\033[1;31mPREDICTIONS WITH CONFIDENCE UNDER {threshold} WILL BE CONVERTED TO 'EMPTY'\033[0m")
  #i create the FEN one row at a time
  for i in range(len(pieces)):
    em_counter = 0
    pieces_row = []
    for j in range(len(pieces[i])):
      #if confidence < threshold then convert the prediction to empty
      if pieces[i][j] == "em" or confidence[i][j]<=threshold:
        pieces_row.append((fen_notation(pieces[i][j]),confidence[i][j].item()))
        chessboard.append(fen_notation(pieces[i][j]))
        em_counter += 1
      else:
        if em_counter != 0:
          #pieces_row.append(str(em_counter))
          fen = fen + str(em_counter)
          em_counter = 0
        pieces_row.append((fen_notation(pieces[i][j]),confidence[i][j].item()))
        chessboard.append(fen_notation(pieces[i][j]))
        #pieces_row.append(fen_notation(pieces[i][j]))
        fen = fen + fen_notation(pieces[i][j])
    if em_counter != 0:
      fen = fen + str(em_counter)
      #pieces_row.append(str(em_counter))
    if i < 7:
      fen = fen + "/"
    #print(pieces_row)

  #print("\nChessboard:\n")
  #print(np.asarray(chessboard).reshape(8,8))
  if additional_info:
    print("\nWhite or black turn?\nw white\nb black")
    fen = fen + space + input() + space
    print("\nWho can castle ?\n- if neither side can castle\nKQkq if both side can castle\nK if white can castle king side\nQ if white can castle queen side\nk if black can castle king side\nq if black can castle queen side")
    fen = fen + input() + space + fixed_conclusion
  #FEN where it's white turn and both can castle either king and queen side
  else: fen = fen + space + "w" + space + "KQkq" + space + fixed_conclusion
  print(f"\nFEN: {fen}\n")
  return fen,chessboard


#Screenshot resize to be exactly of size 800x800
def resize_screenshot_inplace(image_path):
    img = cv2.imread(image_path)
    if img is None:
      print(f"Cannot read image: {image_path}, check the path integrity")
      exit()

    resized_img = cv2.resize(img,(800, 800),interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, resized_img)
    return image_path


#CMD print style for better visualization of the predictions
def print_chessboard_terminal(chessboard):
    symbols = {
        "P": "♙", "p": "♟",
        "N": "♘", "n": "♞",
        "B": "♗", "b": "♝",
        "R": "♖", "r": "♜",
        "Q": "♕", "q": "♛",
        "K": "♔", "k": "♚",
        "E": "·"
    }

    board = np.array(chessboard).reshape(8, 8)

    print("\n  a b c d e f g h")
    for i, row in enumerate(board):
        rank = 8 - i
        print(rank, end=" ")
        for cell in row:
            print(symbols.get(cell, "·"), end=" ")
        print(rank)
    print("  a b c d e f g h\n")


if __name__ == "__main__":
  
    screenshot_path = input("Drag and drop the screenshot here:\n").strip().strip('"')
    model_path = os.path.join(os.path.dirname(__file__), "CNN_bleeding.pth")
    pieces_path = os.path.join(os.path.dirname(__file__), "fen_to_image")

    model = CNN(len(target_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    chessboard = np.array(cells_from_screenshot(resize_screenshot_inplace(screenshot_path)))
    print("IF THE CELLS ARE TOO WEIRDLY CROPPED PLEASE BETTER RESIZE THE SCREENSHOT OR THE CNN WILL HAVE A HARD TIME PREDICTING ON IT\nCTRL + c to end this process")
    pieces_predictions = [[], []]

    with torch.no_grad():
        for X in chessboard:
            X = transform(X)
            X = X.unsqueeze(0).to(device)
            result = model(X)
            prob = torch.softmax(result, dim=1)
            confidence, pred = prob.max(dim=1)

            pieces_predictions[0].append(target_names[pred.item()])
            pieces_predictions[1].append(confidence)

    fen, predicted_chessboard = fen_from_predictions(pieces_predictions,additional_info=True)
    print(np.asarray(predicted_chessboard).reshape(8,8))
    boardImage = fenToImage(fen=fen,
                          squarelength=80,
                          pieceSet=loadPiecesFolder(path=pieces_path, cache=True),
                          darkColor="#D18B47",
                          lightColor="#FFCE9E")
    boardImage.show()
    #CMD doesn't support Unicode characters
    #print_chessboard_terminal(predicted_chessboard)
