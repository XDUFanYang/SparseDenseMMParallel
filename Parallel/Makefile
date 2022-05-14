test:
	g++ -o test dense-denseblock.cpp
	./test
	g++ -o test coo-dense.c
	./test
	g++ pthreadmatrix.cpp -o test -fopenmp -lpthread
	./test 10
	g++ ompcoodense.cpp -o test -fopenmp -lpthread
	./test
