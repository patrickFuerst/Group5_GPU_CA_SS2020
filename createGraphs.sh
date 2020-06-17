#!/bin/sh


num_nodes=(10 1000)
graph_densities=(0.1 0.3 0.5 0.7 0.9 1.0)

low_weight=1
high_weight=100

echo "Creating graphs!"

cd data/graphs

for nodes in "${num_nodes[@]}"; do
	for density in "${graph_densities[@]}"; do

		echo  "Create graph with $nodes nodes and $density density"
		../../release/graphGenerator -p $nodes $density $low_weight $high_weight
		echo "----------------------"
	done 
done

echo "FINISHED creating graphs!"
