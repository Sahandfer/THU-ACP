package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/pprof"
	"strconv"
)

type Edge struct {
	src, dst uint32
}

type Partition struct {
	edgeCount, mirrorEdgeCount, masterCount, mirrorCount int
	mirror, master                                       map[uint32]bool
}

func getReplicationFactor(partitions []Partition) float32 {
	nNodes := 0
	totalNodes := 0
	for _, v := range partitions {
		nNodes += len(v.master)
		totalNodes += len(v.master) + len(v.mirror)
	}

	if totalNodes == 0 {
		for _, v := range partitions {
			nNodes += v.masterCount
			totalNodes += v.masterCount + v.mirrorCount
		}
	}

	return float32(totalNodes) / float32(nNodes)
}

func edge(nMachines int, next chan uint32) {
	partitions := make([]Partition, nMachines)
	for i := range partitions {
		partitions[i].edgeCount = 0
		partitions[i].mirrorEdgeCount = 0
		partitions[i].mirror = make(map[uint32]bool)
		partitions[i].master = make(map[uint32]bool)
	}
	for {
		src, more := <-next
		if !more {
			break
		}
		dst := <-next

		i1 := src % uint32(nMachines)
		i2 := dst % uint32(nMachines)

		// update src info
		partitions[i1].edgeCount++
		partitions[i1].master[src] = true

		// update dst info
		partitions[i2].edgeCount++
		partitions[i2].master[dst] = true

		if i1 != i2 {
			// mirrors and replicated edges
			partitions[i1].mirror[dst] = true
			partitions[i1].mirrorEdgeCount++
			// fmt.Printf("%v %v -> %v\n", src, dst, i1)
			partitions[i2].mirror[src] = true
			partitions[i2].mirrorEdgeCount++
			// fmt.Printf("%v %v -> %v\n", src, dst, i2)
		} else {
			// both are masters, avoid duplicated edge count
			partitions[i1].edgeCount--
			// fmt.Printf("%v %v -> %v\n", src, dst, i1)
		}
	}

	// output
	for i := 0; i < nMachines; i++ {
		fmt.Printf("Partition %v\n%v\n%v\n%v\n%v\n\n", i, len(partitions[i].master), len(partitions[i].master)+len(partitions[i].mirror), partitions[i].mirrorEdgeCount, partitions[i].edgeCount)
		// fmt.Printf("%v %v\n", partitions[i].master, partitions[i].mirror)
	}
	fmt.Printf("Replication factor: %v\n", getReplicationFactor(partitions))
}

func vertex(nMachines int, next chan uint32) {
	partitions := make([]Partition, nMachines)
	for i := range partitions {
		partitions[i].edgeCount = 0
		partitions[i].mirror = make(map[uint32]bool)
		partitions[i].master = make(map[uint32]bool)
	}

	replicated := make(map[uint32][]bool, nMachines)

	counter := 0 // keeps track of the current edge index
	for {
		src, more := <-next
		if !more {
			break
		}
		dst := <-next
		// fmt.Printf("%v %v\n", src, dst)
		// get the machine given the index
		index := counter % nMachines
		counter++

		// count increase
		partitions[index].edgeCount++
		// fmt.Printf("%v %v -> %v\n", src, dst, index)
		i1 := src % uint32(nMachines)
		i2 := dst % uint32(nMachines)

		src_machines, ok := replicated[src]
		if !ok {
			src_machines = make([]bool, nMachines)
			replicated[src] = src_machines
		}

		dst_machines, ok := replicated[dst]
		if !ok {
			dst_machines = make([]bool, nMachines)
			replicated[dst] = dst_machines
		}

		src_machines[i1] = true
		src_machines[index] = true
		dst_machines[i2] = true
		dst_machines[index] = true

	}

	// fmt.Printf("%v\n", replicated)
	for i, machines := range replicated {
		index := i % uint32(nMachines)
		partitions[index].masterCount++
		for m, v := range machines {
			if v && m != int(index) {
				partitions[m].mirrorCount++
			}
		}
	}

	// output
	for i := 0; i < nMachines; i++ {

		fmt.Printf("Partition %v\n%v\n%v\n%v\n\n", i, partitions[i].masterCount, partitions[i].masterCount+partitions[i].mirrorCount, partitions[i].edgeCount)
		// fmt.Printf("%v %v\n", partitions[i].master, partitions[i].mirror)
	}
	fmt.Printf("Replication factor: %v\n", getReplicationFactor(partitions))
}

func greedy(nMachines int, next chan uint32) {
	partitions := make([]Partition, nMachines)
	for i := range partitions {
		partitions[i].edgeCount = 0
		partitions[i].mirror = make(map[uint32]bool)
		partitions[i].master = make(map[uint32]bool)
	}

	// highly efficient array of bools
	replicated := make(map[uint32][]bool)

	vertices := make(map[uint32]uint32)

	for {

		src, more := <-next
		if !more {
			break
		}
		dst := <-next

		// get machines where vertex is replicated
		src_machines, ok := replicated[src]
		if !ok {
			src_machines = make([]bool, nMachines)
			replicated[src] = src_machines

		}

		dst_machines, ok := replicated[dst]
		if !ok {
			dst_machines = make([]bool, nMachines)
			replicated[dst] = dst_machines
		}

		// find intersection
		intersect := false
		src_status := false // if empty
		dst_status := false // if empty
		for i := 0; i < nMachines; i++ {
			if src_machines[i] && dst_machines[i] {
				intersect = true
			}

			if src_machines[i] {
				src_status = true
			}

			if dst_machines[i] {
				dst_status = true
			}

		}

		var cond func(i int) bool

		switch {
		case intersect:
			// case 1
			cond = func(i int) bool {
				return src_machines[i] && dst_machines[i]
			}
		case src_status || dst_status:
			// case 2 && 3
			cond = func(i int) bool {
				return src_machines[i] || dst_machines[i]
			}
		default:
			// case 4
			cond = func(i int) bool {
				return true
			}
		}
		var leastCount uint32 = ^uint32(0)
		machine := -1
		for i := 0; i < nMachines; i++ {
			if cond(i) && partitions[i].edgeCount < int(leastCount) {
				leastCount = uint32(partitions[i].edgeCount)
				machine = i
			}
		}

		// count increase
		partitions[machine].edgeCount++

		if _, ok := vertices[src]; !ok {
			vertices[src] = uint32(machine)
		}

		if _, ok := vertices[dst]; !ok {
			vertices[dst] = uint32(machine)
		}

		// remove them from being mirrors later
		partitions[machine].mirror[src] = true
		partitions[machine].mirror[dst] = true

		if v, ok := replicated[src]; ok {
			v[machine] = true
		} else {
			arr := make([]bool, nMachines)
			arr[machine] = true
			replicated[src] = arr
		}

		if v, ok := replicated[dst]; ok {
			v[machine] = true
		} else {
			arr := make([]bool, nMachines)
			arr[machine] = true
			replicated[dst] = arr
		}

	}

	for key := range vertices {
		partitions[vertices[key]].master[key] = true
		delete(partitions[vertices[key]].mirror, key)
	}

	for _, v := range partitions {
		for key := range v.master {
			if _, ok := v.mirror[key]; ok {
				panic(fmt.Sprintf("Found both master and mirror %v", key))
			}
		}
	}

	// output
	for i := 0; i < nMachines; i++ {
		fmt.Printf("Partition %v\n%v\n%v\n%v\n\n", i, len(partitions[i].master), len(partitions[i].master)+len(partitions[i].mirror), partitions[i].edgeCount)
		// fmt.Printf("%v %v\n", partitions[i].master, partitions[i].mirror)
	}
	fmt.Printf("Replication factor: %v\n", getReplicationFactor(partitions))
}

func hybrid(nMachines int, next chan uint32) {
	args := os.Args
	var threshold int
	if len(args) < 5 {
		threshold = 3
	} else {
		v, err := strconv.Atoi(args[4])
		if err != nil || v < 0 {
			panic("Threshold is not a valid number! Input a positive integer.")
		}
		threshold = v
	}
	fmt.Printf("Running with threshold %v\n", threshold)

	partitions := make([]Partition, nMachines)
	for i := range partitions {
		partitions[i].edgeCount = 0
		partitions[i].mirror = make(map[uint32]bool)
		partitions[i].master = make(map[uint32]bool)
	}

	count := make(map[uint32][]uint32)
	highDegree := make(map[uint32]bool)

	for {
		src, more := <-next
		if !more {
			break
		}
		dst := <-next
		// fmt.Printf("(%v,%v)\n", src, dst)
		// high degree vertex just do the source part
		if _, ok := highDegree[dst]; ok {
			// add vertice to source instead
			index := src % uint32(nMachines)
			partitions[index].master[src] = true
			partitions[index].mirror[dst] = true
			partitions[index].edgeCount++
			// fmt.Printf("%v %v -> %v\n", src, dst, index)

		} else {
			// either don't know enough or is a low vertice
			// edge gets placed in index of target
			index := dst % uint32(nMachines)

			partitions[index].edgeCount++
			partitions[index].master[dst] = true
			// fmt.Printf("%v %v -> %v\n", src, dst, index)
			partitions[src%uint32(nMachines)].master[src] = true

			// handle high order vertices
			if _, ok := count[dst]; !ok {
				count[dst] = make([]uint32, 0)
			}

			count[dst] = append(count[dst], src)

			if len(count[dst]) > threshold {
				// fmt.Printf("Vertex %v is high degree\n", dst)
				// become a high degree vertice
				highDegree[dst] = true
				// fmt.Printf("Redo %v\n", dst)
				// move edges into other machines
				partitions[index].edgeCount = partitions[index].edgeCount - len(count[dst])
				for _, src := range count[dst] {
					new_index := src % uint32(nMachines)
					partitions[new_index].edgeCount++
					// fmt.Printf("%v %v -> %v\n", src, dst, new_index)
					partitions[new_index].master[src] = true
					if new_index != index {
						partitions[new_index].mirror[dst] = true
					}
				}

				delete(count, dst)
			}
		}

	}

	// fmt.Printf("%v\n", count)
	// need to add missing mirrors from low degree vertices
	for dst, srcs := range count {
		index := dst % uint32(nMachines)
		for _, src := range srcs {
			if src%uint32(nMachines) != index {
				partitions[index].mirror[src] = true
			}
		}
	}
	fmt.Printf("High degree: %v\n", len(highDegree))
	// output
	for i := 0; i < nMachines; i++ {
		fmt.Printf("Partition %v\n%v\n%v\n%v\n\n", i, len(partitions[i].master), len(partitions[i].master)+len(partitions[i].mirror), partitions[i].edgeCount)
		// fmt.Printf("%v %v\n", partitions[i].master, partitions[i].mirror)
	}
	fmt.Printf("Replication factor: %v\n", getReplicationFactor(partitions))
}

// var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

func main() {
	args := os.Args
	flag.Parse()

	if *memprofile != "" {
		args = args[1:]
	}

	if len(args) < 4 {
		fmt.Printf("Usage: %v <edge|vertex|hybrid|greedy> <#machines> <filename>\n", args[0])
		os.Exit(1)
	}

	nMachines, err := strconv.Atoi(args[2])
	if err != nil {
		panic(err)
	}

	fmt.Printf("Split method = %v, #machines = %v\n", args[1], nMachines)

	fd, err := os.Open(args[3])
	if err != nil {
		panic(err)
	}

	// read the file and put each vertex found into the channel
	next := make(chan uint32, 1024)
	go func() {
		r := bufio.NewReader(fd)
		var buf [1024]byte
		stop := false
		for !stop {
			amount, err := io.ReadFull(r, buf[:])

			if err != nil {
				if err != io.EOF && err != io.ErrUnexpectedEOF {
					panic(err)
				}
				stop = true
			}

			for i := 0; i < amount/4; i++ {
				next <- binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
			}

		}
		close(next)
	}()

	var function func(int, chan uint32)
	switch args[1] {
	case "edge":
		function = edge
	case "vertex":
		function = vertex
	case "hybrid":
		function = hybrid
	case "greedy":
		function = greedy
	default:
		panic("Unrecognized splitting method. Available: edge, vertex, hybrid, greedy")

	}

	function(nMachines, next)
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.WriteHeapProfile(f)
		f.Close()
		return
	}
}
