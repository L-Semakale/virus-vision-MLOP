def log(message: str):
    with open("logs.txt", "a") as f:
        f.write(message + "\n")
