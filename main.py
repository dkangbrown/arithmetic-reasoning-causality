import sys
sys.path.append("src")
import _util

def main():
    print("Hello from arithmetic-reasoning-causality!")
    model, tokenizer = _util.load_OSS()
    print(model)
    print(tokenizer)


if __name__ == "__main__":
    main()
