#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <algorithm>

using namespace std;

struct Partition
{
    int num_edges;
    int num_vertices;
    int num_rep_edges;
    int num_masters;
};

// This function prints out the results
void print_res(string file_name, vector<Partition> partitions, bool print_rep_edges = false)
{
    ofstream output_file;
    output_file.open(file_name);

    string output = "";
    auto total_masters = 0;
    auto total_vertices = 0;

    for (int i = 0; i < partitions.size(); i++)
    {
        output += "Partition " + to_string(i) + "\n";
        output += to_string(partitions[i].num_masters) + "\n";
        output += to_string(partitions[i].num_vertices) + "\n";
        if (print_rep_edges)
            output += to_string(partitions[i].num_rep_edges) + "\n";
        output += to_string(partitions[i].num_edges) + "\n";
        output += "\n";

        total_masters += partitions[i].num_masters;
        total_vertices += partitions[i].num_vertices;
    }

    float rep_factor = (float)total_vertices / (float)total_masters;
    output += "Replication Factor: " + to_string(rep_factor) + "\n";

    cout << output;
    output_file << output;
    output_file.close();
}

// This function finds the max degree vertex in the dataset
int find_max_vertice(char *input_file)
{
    FILE *fin = fopen(input_file, "rb");
    int max_vertice = 0;
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        max_vertice = max({max_vertice, src_id, dst_id});
    }
    fclose(fin);

    return max_vertice;
}

// This function finds the in-degree of destination vertices
vector<int> find_degree(char *input_file, int max_vertice)
{
    FILE *fin = fopen(input_file, "rb");
    vector<int> degree(max_vertice + 1, 0);
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        degree[dst_id]++;
    }
    fclose(fin);

    return degree;
}

vector<Partition> balanced_edge_cut(char *input_file, int num_partitions)
{
    vector<Partition> partitions;
    for (int i = 0; i < num_partitions; i++)
    {
        Partition p;
        p.num_edges = 0;
        p.num_vertices = 0;
        p.num_rep_edges = 0;
        p.num_masters = 0;
        partitions.push_back(p);
    }

    vector<set<int> > graph;
    int max_vertice = find_max_vertice(input_file);
    for (int i = 0; i <= max_vertice; i++)
    {
        graph.push_back(set<int>());
    }

    FILE *fin = fopen(input_file, "rb");
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        // Find corresponding partition
        int src_par = src_id % num_partitions;
        int dst_par = dst_id % num_partitions;
        // Add an edge to each partition
        partitions[src_par].num_edges++;
        partitions[dst_par].num_edges++;

        // Add partition to the map for each vertice
        graph[src_id].insert(src_par);
        graph[dst_id].insert(dst_par);

        // Duplicate edge
        if (src_par == dst_par)
            partitions[src_par].num_edges--;
        else
        {
            graph[src_id].insert(dst_par);
            graph[dst_id].insert(src_par);
            partitions[src_par].num_rep_edges++;
            partitions[dst_par].num_rep_edges++;
        }
    }
    fclose(fin);

    for (int i = 0; i <= max_vertice; i++)
    {
        if (graph[i].size())
        {
            partitions[i % num_partitions].num_masters++;
            for (auto it = graph[i].begin(); it != graph[i].end(); ++it)
                partitions[*it].num_vertices++;
        }
    }

    return partitions;
}

vector<Partition> balanced_vertex_cut(char *input_file, int num_partitions)
{
    vector<Partition> partitions;
    for (int i = 0; i < num_partitions; i++)
    {
        Partition p;
        p.num_edges = 0;
        p.num_vertices = 0;
        p.num_rep_edges = 0;
        p.num_masters = 0;
        partitions.push_back(p);
    }

    vector<set<int> > graph;
    int max_vertice = find_max_vertice(input_file);
    for (int i = 0; i <= max_vertice; i++)
    {
        graph.push_back(set<int>());
    }

    int edge_id = 0;
    FILE *fin = fopen(input_file, "rb");
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        // Find corresponding partition
        int src_par = src_id % num_partitions;
        int dst_par = dst_id % num_partitions;
        // Get edge partition
        int edge_par = edge_id % num_partitions;
        partitions[edge_id].num_edges++;
        // Add edge and source partition for source vertice
        graph[src_id].insert(src_par);
        graph[src_id].insert(edge_id);
        // Add edge and destination partition for destination vertice
        graph[dst_id].insert(dst_par);
        graph[dst_id].insert(edge_id);
        // Increase id to point to next edge
        edge_id++;
        // Reset ids
        if (edge_id % num_partitions == 0) edge_id = 0;
    }
    fclose(fin);

    for (int i = 0; i <= max_vertice; i++)
    {
        if (graph[i].size())
        {
            partitions[i % num_partitions].num_masters++;
            for (auto it = graph[i].begin(); it != graph[i].end(); ++it)
                partitions[*it].num_vertices++;
        }
    }

    return partitions;
}


vector<Partition> greedy_vertex_cut(char *input_file, int num_partitions)
{
    vector<Partition> partitions;
    for (int i = 0; i < num_partitions; i++)
    {
        Partition p;
        p.num_edges = 0;
        p.num_vertices = 0;
        p.num_rep_edges = 0;
        p.num_masters = 0;
        partitions.push_back(p);
    }
    vector<set<int> > graph;
    int max_vertice = find_max_vertice(input_file);
    for (int i = 0; i <= max_vertice; i++)
    {
        graph.push_back(set<int>());
    }

    FILE *fin = fopen(input_file, "rb");
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        // Find corresponding partitions
        int src_par = src_id % num_partitions;
        int dst_par = dst_id % num_partitions;
        // Find intersection
        vector<int> intersection;
        set_intersection(graph[src_id].begin(), graph[src_id].end(), graph[dst_id].begin(), graph[dst_id].end(), back_inserter(intersection));
        // Case 1
        if (intersection.size()){
            int intersect = intersection[0];
            partitions[intersect].num_edges++;
        }
        // Cases 2 and 3
        else if (!graph[src_id].empty() || !graph[dst_id].empty()){
            int min = INT_MAX;
            int min_idx = -1;
            bool from_src = true;
            // Find minimum partition from source
            vector<int> v_src(graph[src_id].begin(), graph[src_id].end());
            for (auto it = v_src.begin(); it != v_src.end(); ++it)
            {
                if (partitions[*it].num_edges < min)
                {
                    min = partitions[*it].num_edges;
                    min_idx = *it;
                }
            }
            // Find minimum partition from destination
            vector<int> v_dst(graph[dst_id].begin(), graph[dst_id].end());
            for (auto it = v_dst.begin(); it != v_dst.end(); ++it)
            {
                if (partitions[*it].num_edges < min)
                {
                    min = partitions[*it].num_edges;
                    min_idx = *it;
                    from_src = false;
                }
            }
            // Add edge to minimum partition
            partitions[min_idx].num_edges++;
            // If minimum partition is from source, add destination partition
            if (from_src)
            {
                graph[dst_id].insert(min_idx);
            }
            // If minimum partition is from destination, add source partition
            else
            {
                graph[src_id].insert(min_idx);
            }
        }
        // Case 4
        else {
            int min = INT_MAX;
            int min_idx = -1;
            // Find least loaded partition
            for (int i = 0; i < partitions.size(); i++)
            {
                if (partitions[i].num_edges < min)
                {
                    min = partitions[i].num_edges;
                    min_idx = i;
                }
            }
            // Add edge to least loaded partition
            partitions[min_idx].num_edges++;
            // Add vertices to least loaded partition
            graph[src_id].insert(min_idx);
            graph[dst_id].insert(min_idx);
        }
    }
    fclose(fin);
    for (int i = 0; i <= max_vertice; i++)
    {
        if (graph[i].size())
        {
            partitions[*graph[i].begin()].num_masters++;
            for (auto it = graph[i].begin(); it != graph[i].end(); ++it)
                partitions[*it].num_vertices++;
        }
    }
    return partitions;
}

vector<Partition> balanced_hybrid_cut(char *input_file, int num_partitions, int threshold)
{
    vector<Partition> partitions;
    for (int i = 0; i < num_partitions; i++)
    {
        Partition p;
        p.num_edges = 0;
        p.num_vertices = 0;
        p.num_rep_edges = 0;
        p.num_masters = 0;
        partitions.push_back(p);
    }

    vector<set<int> > graph;
    int max_vertice = find_max_vertice(input_file);
    vector<int> in_degrees = find_degree(input_file, max_vertice);
    for (int i = 0; i <= max_vertice; i++)
    {
        graph.push_back(set<int>());
    }

    FILE *fin = fopen(input_file, "rb");
    while (true)
    {
        // Get vertices ID
        int src_id, dst_id;
        if (fread(&src_id, sizeof(src_id), 1, fin) == 0)
            break;
        if (fread(&dst_id, sizeof(dst_id), 1, fin) == 0)
            break;
        // Find corresponding partitions
        int src_par = src_id % num_partitions;
        int dst_par = dst_id % num_partitions;
        // Low-degree vertice
        if (in_degrees[dst_id] <= threshold)
        {
            graph[src_id].insert(dst_par);
            graph[dst_id].insert(dst_par);
            partitions[dst_par].num_edges++;
        }
        // High-degree vertice
        else
        {
            graph[src_id].insert(src_par);
            graph[dst_id].insert(src_par);
            partitions[src_par].num_edges++;
        }
    }
    fclose(fin);
    for (int i = 0; i <= max_vertice; i++)
    {
        if (graph[i].size())
        {
            partitions[i % num_partitions].num_masters++;
            for (auto it = graph[i].begin(); it != graph[i].end(); ++it)
                partitions[*it].num_vertices++;
        }
    }

    return partitions;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: [executable] [input_file] [partition type] [num partitions] [threshold]\n");
        exit(-1);
    }

    string input_file = argv[1];
    string part_type = argv[2];
    int num_partitions = atoi(argv[3]);
    int threshold = atoi(argv[4]);

    // map<int, int> in_degrees;
    map<string, int> type_map = {{"edge", 0}, {"vertex", 1}, {"greedy", 2}, {"hybrid", 3}};
    int cut_type = type_map[part_type];

    vector<Partition> res;
    switch (cut_type)
    {
    // Edge-cut
    case 0:
        res = balanced_edge_cut(argv[1], num_partitions);
        break;
    // Vertex-cut
    case 1:
        res = balanced_vertex_cut(argv[1], num_partitions);
        break;
    case 2:
        res = greedy_vertex_cut(argv[1], num_partitions);
        break;
    case 3:
        res = balanced_hybrid_cut(argv[1], num_partitions, threshold);
        break;
    default:
        cout << "Invalid partitioning type" << endl;
    }

    string file_name = "output//" + input_file.substr(5, input_file.size() - 11) + "_" + part_type + "_" + to_string(num_partitions) + "_machines.txt";
    print_res(file_name, res, !cut_type);
    return 0;
}