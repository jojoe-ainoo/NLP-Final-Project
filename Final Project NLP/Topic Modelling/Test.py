



if __name__ == "__main__":
    
    if sys.argv[1] == "Topic" and sys.argv[2] == "u":
        naiveUnormalized("trainer.txt",sys.argv[3])
    elif sys.argv[1] == "nb" and sys.argv[2] == "n":
        naiveNormalized("trainer.txt",sys.argv[3])

    
    
else:
    print("Try again and add file name")

