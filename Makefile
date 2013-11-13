all: learn

learn: ./train/learn.cc
	g++ -O3 -o ./bin/learn ./train/learn.cc

clean:
	/bin/rm ./binlearn




