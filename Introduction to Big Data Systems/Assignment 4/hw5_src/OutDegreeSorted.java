import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class OutDegreeSorted {

	// Mapper - out degree
	public static class OutDegreeMapper
			extends Mapper<Object, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1); // value
		private Text node = new Text(); // key

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString(), "\\n"); // split by lines
			while (itr.hasMoreTokens()) {
				node.set(itr.nextToken().split(" ")[1]); // get u from (a, u, v, w)
				context.write(node, one); // write the (key, value) pair to context
			}
		}
	}

	// Reducer - out degree
	public static class OutDegreeReducer
			extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable(); // out-degree

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {

			int sum = 0;
			// count the number of occurrences
			for (IntWritable val : values) {
				sum += val.get(); // set sum as value for the node
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	// Mapper - sorter
	public static class OutDegreeSortMapper
			extends Mapper<Object, Text, IntWritable, Text> {

		private Text node = new Text(); // key
		private IntWritable outdegree = new IntWritable(); // value

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString(), "\\n"); // split by lines
			while (itr.hasMoreTokens()) {
				String[] line = itr.nextToken().split("\\t");
				node.set(line[0]); // get u from (u, out_degree)
				outdegree.set(Integer.parseInt(line[1])); // get out_degree from (u, out_degree)
				context.write(outdegree, node); // write the (value, key) to reverse the pair
			}
		}
	}

	// Reducer - sorter
	public static class OutDegreeSortReducer
			extends Reducer<IntWritable, Text, Text, IntWritable> {
		int count = 0; // Number of nodes already saved

		public void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			// the number of top-n biggest nodes
			int print_len = Integer.parseInt(context.getConfiguration().get("print_len"));
			for (Text val : values) {
				if (count >= print_len)
					break;
				context.write(val, key);
				count++;
			}
		}
	}

	public static class OutDegreeSortComparator extends WritableComparator {
		public OutDegreeSortComparator() {
			super(IntWritable.class, true);
		}

		public int compare(WritableComparable wc1, WritableComparable wc2) {
			IntWritable key1 = (IntWritable) wc1;
			IntWritable key2 = (IntWritable) wc2;
			return -1 * key1.compareTo(key2);
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		int n = otherArgs.length;
		if (n < 3) {
			System.err.println("Usage: outdegree <in> <out> <print_len>");
			System.exit(2);
		}
		Job job_1 = new Job(conf, "outdegree");
		job_1.setJarByClass(OutDegreeSorted.class);
		job_1.setMapperClass(OutDegreeMapper.class);
		job_1.setCombinerClass(OutDegreeReducer.class);
		job_1.setReducerClass(OutDegreeReducer.class);
		job_1.setNumReduceTasks(1);
		job_1.setOutputKeyClass(Text.class);
		job_1.setOutputValueClass(IntWritable.class);

		FileInputFormat.addInputPath(job_1, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job_1, new Path(otherArgs[n - 2]));
		job_1.waitForCompletion(true);

		conf.set("print_len", otherArgs[n - 1]);
		Job job_2 = new Job(conf, "sort");
		job_2.setJarByClass(OutDegreeSorted.class);
		job_2.setMapperClass(OutDegreeSortMapper.class);
		job_2.setSortComparatorClass(OutDegreeSortComparator.class);
		job_2.setReducerClass(OutDegreeSortReducer.class);
		job_2.setMapOutputKeyClass(IntWritable.class);
		job_2.setMapOutputValueClass(Text.class);
		job_2.setOutputKeyClass(Text.class);
		job_2.setOutputValueClass(IntWritable.class);

		FileInputFormat.addInputPath(job_2, new Path(otherArgs[n - 2]));
		FileOutputFormat.setOutputPath(job_2, new Path(otherArgs[n - 2] + "_sort"));
		System.exit(job_2.waitForCompletion(true) ? 0 : 1);
	}
}
