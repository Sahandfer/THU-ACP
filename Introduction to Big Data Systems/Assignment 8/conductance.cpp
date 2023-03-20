#include "core/graph.hpp"
#include <algorithm>

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "usage: pagerank [path] [iterations] [memory budget in GB]\n");
		exit(-1);
	}
	string path = argv[1];
	long memory_bytes = (argc >= 3) ? atol(argv[2]) * 1024l * 1024l * 1024l : 8l * 1024l * 1024l * 1024l;

	Graph graph(path);
	graph.set_memory_bytes(memory_bytes);

	long vertex_data_bytes = (long)graph.vertices * (sizeof(VertexId) + sizeof(float) + sizeof(float));
	graph.set_vertex_data_bytes(vertex_data_bytes);

	double begin_time = get_time();

	int black_num = 0;
	int red_num = 0;
	int crossover_num = 0;

	graph.stream_edges<VertexId>(
		[&](Edge &e)
		{
			// Crossover edge
			if ((e.source & 1) != (e.target & 1))
			{
				write_add(&crossover_num, 1);
			}
			// Red edge
			if (e.source & 1)
			{
				write_add(&red_num, 1);
			}
			// Black edge
			else
			{
				write_add(&black_num, 1);
			}
			return 0;
		},
		nullptr, 0, 0);

	double end_time = get_time();
	float conductance = (float)crossover_num / (float)min(red_num, black_num);

	printf("Conductance is calculated as %.2f - Process took %.2f seconds\n", conductance, end_time - begin_time);
	printf("#Crossover: %d - #Reds: %d - #Blacks: %d\n", crossover_num, red_num, black_num);
}