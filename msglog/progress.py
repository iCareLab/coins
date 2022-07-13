import colorama

def progress(progress, total, color=colorama.Fore.YELLOW):
    percent = 100 * (progress / float(total))
    bar = '#' * int(percent) + '-' * (100 - int(percent))
    print(color + f"\r|{bar}| {percent:.2f}%", end="\r")
    if float(progress) >= float(total):
        print(colorama.Fore.GREEN + f"\r|{bar}| {percent:.2f}%", end="\r")
        print(colorama.Fore.RESET)

def traning(progress, total, epoch=0, loss=0.0, color=colorama.Fore.YELLOW):
    percent = 100 * (progress / float(total))
    bar = '#' * int(percent) + '-' * (100 - int(percent))
    print(color + f"\r|{bar}| {percent:.2f}%, Epoch:{epoch+1}, loss:{loss:1.5f}", end="\r")
    if float(progress) >= float(total):
        print(colorama.Fore.GREEN + f"\r|{bar}| {percent:.2f}%, Epoch:{epoch+1}, loss:{loss:1.5f}", end="\r")
        print(colorama.Fore.RESET)

if __name__ == '__main__':
    print("TEST for Progress Bar")
    for i in range(100):
        progress(i+1, 100)

    print("TEST END")
