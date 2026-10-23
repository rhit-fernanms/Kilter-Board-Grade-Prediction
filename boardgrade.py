import sys

difficulties = {
  10: "4a",
  11: "4b",
  12: "4c",
  13: "5a",
  14: "5b",
  15: "5c",
  16: "6a",
  17: "6a+",
  18: "6b",
  19: "6b+",
  20: "6c",
  21: "6c+",
  22: "7a",
  23: "7a+",
  24: "7b",
  25: "7b+",
  26: "7c",
  27: "7c+",
  28: "8a",
  29: "8a+",
  30: "8b",
  31: "8b+",
  32: "8c",
  33: "8c+",
}

if len(sys.argv) <= 1:
  print("usage: boardgrade.py [grade number] to get the grade associated with a number")
else:
  inputNum = int(sys.argv[1])
  print(f"{difficulties[inputNum]}")
