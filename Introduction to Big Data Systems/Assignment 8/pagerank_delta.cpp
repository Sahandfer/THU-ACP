/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "core/graph.hpp"

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "usage: pagerank [path] [iterations] [memory budget in GB]\n");
		exit(-1);
	}
	std::string path = argv[1];
	int iterations = atoi(argv[2]);
	long memory_bytes = (argc >= 4) ? atol(argv[3]) * 1024l * 1024l * 1024l : 8l * 1024l * 1024l * 1024l;
	float threshold = atof(argv[4]);

	Graph graph(path);
	graph.set_memory_bytes(memory_bytes);
	int num_vertices = graph.vertices;

	BigVector<VertexId> degree(graph.path + "/degree", num_vertices);
	BigVector<float> pagerank(graph.path + "/pagerank", num_vertices);
	BigVector<float> sum(graph.path + "/sum", num_vertices);
	BigVector<float> delta(graph.path + "/delta", num_vertices);

	long vertex_data_bytes = (long)num_vertices * (sizeof(VertexId) + sizeof(float) + sizeof(float));
	graph.set_vertex_data_bytes(vertex_data_bytes);

	double begin_time = get_time();

	degree.fill(0);
	graph.stream_edges<VertexId>(
		[&](Edge &e)
		{
			write_add(&degree[e.source], 1);
			return 0;
		},
		nullptr, 0, 0);
	printf("degree calculation used %.2f seconds\n", get_time() - begin_time);
	fflush(stdout);

	graph.hint(pagerank, sum, delta);
	graph.stream_vertices<VertexId>(
		[&](VertexId i)
		{
			pagerank[i] = 1.f / float(num_vertices);
			sum[i] = 0;
			delta[i] = 1.f;
			return 0;
		},
		nullptr, 0,
		[&](std::pair<VertexId, VertexId> vid_range)
		{
			pagerank.load(vid_range.first, vid_range.second);
			sum.load(vid_range.first, vid_range.second);
			delta.load(vid_range.first, vid_range.second);
		},
		[&](std::pair<VertexId, VertexId> vid_range)
		{
			pagerank.save();
			sum.save();
			delta.save();
		});

	for (int iter = 0; iter < iterations; iter++)
	{
		graph.hint(pagerank, delta);
		graph.stream_edges<VertexId>(
			[&](Edge &e)
			{
				// Calculate delta for edge
				write_add(&sum[e.target], delta[e.source] / degree[e.source]);
				return 0;
			},
			nullptr, 0, 1,
			[&](std::pair<VertexId, VertexId> source_vid_range)
			{
				delta.lock(source_vid_range.first, source_vid_range.second);
			},
			[&](std::pair<VertexId, VertexId> source_vid_range)
			{
				delta.unlock(source_vid_range.first, source_vid_range.second);
			});
		graph.hint(pagerank, sum, delta);
		graph.stream_vertices<float>(
			[&](VertexId i)
			{
				delta[i] = 0.85f * sum[i];
				if ((delta[i]/pagerank[i])> threshold){
					pagerank[i] += delta[i];
				}
				sum[i] = 0;
				return 0;
			},
			nullptr, 0,
			[&](std::pair<VertexId, VertexId> vid_range)
			{
				pagerank.load(vid_range.first, vid_range.second);
				sum.load(vid_range.first, vid_range.second);
				delta.load(vid_range.first, vid_range.second);
			},
			[&](std::pair<VertexId, VertexId> vid_range)
			{
				pagerank.save();
				sum.save();
				delta.save();
			});
	}

	double end_time = get_time();
	// Find vertice with maximum rank
	float max_val = -1;
	int max_idx = -1;
	for (int i = 0; i < num_vertices; i++)
	{
		if (pagerank[i] > max_val)
		{
			max_val = pagerank[i];
			max_idx = i;
		}
	}
	printf("%d iterations of pagerank took %.2f seconds\n", iterations, end_time - begin_time);
	printf("Highest Rank Vertice - Index %d - Value %f\n", max_idx, max_val);
}
