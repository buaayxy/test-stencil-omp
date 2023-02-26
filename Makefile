all: omp_o3  big_omp_o3  baseline
.PHONY: all
OPTIMIZATION_LEVEL = -O1
omp_o3:stencil_omp.cpp
	g++ -fopenmp $(OPTIMIZATION_LEVEL) -o omp_o3 stencil_omp.cpp

big_omp_o3:stencil_omp_big_loop.cpp
	g++ -fopenmp $(OPTIMIZATION_LEVEL) -o big_omp_o3 stencil_omp_big_loop.cpp

baseline:omp_baseline.cpp
	g++ -o baseline omp_baseline.cpp -fopenmp $(OPTIMIZATION_LEVEL)

clean:
	rm -f omp_o3 omp_o1 big_omp_o3 big_omp_o1 baseline