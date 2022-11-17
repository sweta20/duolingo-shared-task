import unicodedata
import sys

def main():
	filename = sys.argv[1]
	count = 0
	with open(filename, encoding="utf-8") as f:
		with open(filename + ".norm" , "w" , encoding="utf-8") as f1:
			for line in f:
				count+=1
				f1.write(unicodedata.normalize('NFKC', line))
	print("Number of lines: ", str(count))

if __name__ == '__main__':
	main()