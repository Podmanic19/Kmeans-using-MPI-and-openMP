#include <mpi.h> 
#include <omp.h>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <set>
#include <chrono>
#include <algorithm>
#include <random>
#include <thread>

struct Point {

	int master_id = -1;
	double x = 0;
	double y = 0;
	double z = 0;

};

struct Cluster {

	int id = -1;
	double x = 0;
	double y = 0;
	double z = 0;
	double WCSS = -1;
	short changed = 1;

};


std::ifstream open_file();

std::pair<int, std::vector<Point>> read_file(std::ifstream& ifs);

std::vector<Cluster> select_random_centres(std::vector<Point>& points, int number);
std::vector<std::vector<Point>> sort_points(const std::vector<Point>& points, int cluster_num);

double distance_to(const Point& first, const Point& second);

void print_cluster(Cluster c);
void print_cluster(Cluster cluster);
void print_point_coords(Point p);
void print_point(Point p);
void reassign_points_to_clusts(std::vector<Point>& points, std::vector<Cluster>& clusters);
void recalculate_centre(const std::vector<Point>& points, std::vector<Cluster>& clusters);
void calculate_WCSS(std::vector<Cluster>& clusters, const std::vector<Point>& points);


double distance_to(const Point& first, const Point& second) {

	double distance = 0;

	distance = pow((first.x - second.x), 2) + pow((first.y - second.y), 2) + pow((first.z - second.z), 2);


	return sqrt(distance);

}

void recalculate_centre(const std::vector<Point>& points, std::vector<Cluster>& clusters) {

	for (Cluster& cluster : clusters) {
		double lowest_distance = -1;
		int i = 0;
		int j = 0;
		double local_lowest_distance;
		Point* local_centre;
		#pragma omp parallel default(none) private(i, j, local_lowest_distance, local_centre) shared(lowest_distance, cluster, points)
		{
			local_lowest_distance = -1;
			local_centre = nullptr;
			#pragma omp for
			for (i = 0; i < points.size(); i++) {
				Point first = points[i];
				if (first.master_id != cluster.id) continue;
				double distanceSum = 0;
				for (j = 0; j < points.size(); j++) {
					Point second = points[j];
					if (&first == &second) continue;
					if (second.master_id != first.master_id) continue;
					distanceSum += distance_to(first, second);			// sum the distance between first point and all the others 
				}
				if (local_lowest_distance < 0 || local_lowest_distance > distanceSum) {
					if (local_centre == nullptr)
						local_centre = new Point;
					local_lowest_distance = distanceSum;
					*local_centre = first;
				}
			}
			#pragma omp critical
			{
				if (local_centre != nullptr) {
					if (lowest_distance < 0 || local_lowest_distance < lowest_distance) {
						lowest_distance = local_lowest_distance;
						cluster.x = local_centre->x;
						cluster.y = local_centre->y;
						cluster.z = local_centre->z;
					}
				}
			}
			if (local_centre != nullptr) delete local_centre;
		}
	}

}

void calculate_WCSS(std::vector<Cluster>& clusters, const std::vector<Point>& points) {

	double wcss = 0;

	for (Cluster& cluster : clusters) {
		#pragma omp parallel for reduction(+:wcss)
		for (int i = 0; i < points.size(); i++) {
			const Point& point = points[i];
			if (point.master_id != cluster.id) continue;
			Point p;
			p.x = cluster.x;
			p.y = cluster.y;
			p.z = cluster.z;
			wcss += pow(distance_to(p, point), 2);
		}
		cluster.WCSS = wcss;
	}
}

std::ifstream open_file() {

	std::ifstream input;
	std::string path;

	std::cout << "Enter path to file from which points should be read." << std::endl;
	std::cin >> path;

	input.open(path);

	if (!input.is_open()) {
		throw "Failed to open specified file.";
	}

	return input;
}

std::pair<int, std::vector<Point>> read_file(std::ifstream& ifs) {

	std::pair<int, std::vector<Point>> input;
	std::string line;
	short dimensions_num = 0;

	std::getline(ifs, line);
	input.first = std::stoi(line, nullptr, 10);

	std::getline(ifs, line);
	dimensions_num = std::stoi(line, nullptr, 10);


	while (std::getline(ifs, line)) {

		Point p;
		std::stringstream ss(line);

		ss >> p.x;
		ss >> p.y;
		ss >> p.z;

		input.second.push_back(p);

	}

	return input;

}

bool compare_points(const Point& first, const Point& second) {

	return (first.x == second.x && first.y == second.y && first.z == second.z);

}

std::vector<Point> generate_first_points(int number) {

	std::vector<Point>points;
	points.resize(number);

	double lower_bound = 0;
	double upper_bound = 1000;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < number; i++) {
		bool add = true;
		Point p;
		p.master_id = -1;
		p.x = unif(gen);
		p.y = unif(gen);
		p.z = unif(gen);
		for (Point& other : points) {
			if (compare_points(p, other)) {
				add = false;
				i--;
			}
		}
		if (add)	points[i] = p;
	}

	return points;

}

std::vector<Point> generate_random_points(const std::vector<Point>& starting_points, int number) {

	std::vector<Point> points;
	points.resize(number);

	int lower_bound = 0;
	int upper_bound = starting_points.size();

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_int_distribution<int> unif(lower_bound, upper_bound - 1);
	std::default_random_engine uniform_int;
	std::default_random_engine generator;

	for (unsigned int i = 0; i < number; i++) {
		bool add = true;
		int index = unif(uniform_int);

		const Point& starting = starting_points[index];

		std::normal_distribution<double> distribution_x(starting.x, 200);
		std::normal_distribution<double> distribution_y(starting.y, 200);
		std::normal_distribution<double> distribution_z(starting.z, 200);

		Point p;

		p.x = distribution_x(gen);
		p.y = distribution_y(gen);
		p.z = distribution_z(gen);

		for (Point& other : points) {
			if (compare_points(p, other)) {
				add = false;
				i--;
				break;
			}
		}
		if (add) points[i] = p;

	}

	std::vector<Point> allpoints;

	allpoints.reserve(points.size() + starting_points.size());

	allpoints.insert(allpoints.end(), starting_points.begin(), starting_points.end());
	allpoints.insert(allpoints.end(), points.begin(), points.end());

	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(allpoints), std::end(allpoints), rng);


	return allpoints;

}

std::vector<Cluster> select_random_centres(std::vector<Point>& points, int number) {

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, points.size() - 1);

	std::vector<Cluster> clusters;
	clusters.resize(number);
	std::set<int> chosen;
	clusters.reserve(number);

	for (int i = 0; i < number; i++) {
		Cluster cluster;
		int index = dist(rng);
		if (chosen.find(index) == chosen.end()) {
			chosen.insert(index);
			cluster.x = points[index].x;
			cluster.y = points[index].y;
			cluster.z = points[index].z;
			cluster.id = i;
			points[index].master_id = i;
			clusters[i] = cluster;
		}
		else i--;
	}

	return clusters;

}

void reassign_points_to_clusts(std::vector<Point>& points, std::vector<Cluster>& clusters) {

	#pragma omp parallel for
	for (int i = 0; i < points.size(); i++) {
		Point& p = points[i];
		const int starting_master_id = p.master_id;
		int next_master_id = p.master_id;
		double min_distance;
		if (p.master_id > -1) {
			Point c;
			c.x = clusters[p.master_id].x;
			c.y = clusters[p.master_id].y;
			c.z = clusters[p.master_id].z;
			min_distance = distance_to(p, c);					// calculate distance to current cluster centre
		}
		else {
			min_distance = -1;
		}
		for (int j = 0; j < clusters.size(); j++) {
			if (j == starting_master_id) continue;
			Point c;
			c.x = clusters[j].x;
			c.y = clusters[j].y;
			c.z = clusters[j].z;
			double distance = distance_to(p, c);							// calculate distance to other clusters
			if (distance < min_distance || min_distance < 0) {
				min_distance = distance;
				next_master_id = j;
			}
		}
		p.master_id = next_master_id;
		if (starting_master_id != -1) {
			#pragma omp critical
			{
				if (starting_master_id != next_master_id) clusters[starting_master_id].changed = 1;
			}
		}
	}

}

short clusters_changed(short* cluster_changed, std::vector<Cluster>& clusters) {

	short changed = 0;

	for (unsigned int i = 0; i < clusters.size(); i++) {
		Cluster& cluster = clusters[i];
		if (cluster_changed[i] > 0) changed = 1;
		cluster.changed = 0;
		cluster_changed[i] = 0;
	}

	return changed;

}

void print_points(const std::vector<Point>& points) {

	for (const Point& p : points) {
		print_point(p);
	}

}

void print_clusters(const std::vector<Cluster>& clusters) {

	for (const Cluster& c : clusters) {
		print_cluster(c);
	}

}

void print_point_coords(Point p) {

	std::cout << p.x << " " << p.y << " " << p.z << " ";
	std::cout << std::endl;

}

void print_point(Point p) {

	std::cout << "Master " << p.master_id << " ";
	print_point_coords(p);

}

void print_cluster(Cluster c) {

	std::cout << "ID: " << c.id << std::endl;
	std::cout << "Centre:  " << c.x << " " << c.y << " " << c.z << std::endl;
	std::cout << "WCSS: " << c.WCSS << std::endl;
	std::cout << "Changed: " << c.changed << std::endl << std::endl;


}

std::vector<std::vector<Point>> sort_points(const std::vector<Point>& points, int cluster_num) {

	std::vector<std::vector<Point>> cluster_points;
	cluster_points.resize(cluster_num);

	for (const Point& p : points) {
		cluster_points[p.master_id].push_back(p);
	}

	return cluster_points;

}

void start(int* cluster_num, std::vector<Point>& points) {

	std::cout << "Press 1 for file input, press 2 for random point generation." << std::endl;
	int x = 0;

	std::cin >> x;

	if (x == 1) {
		std::ifstream ifs = open_file();
		std::pair<int, std::vector<Point>> params = read_file(ifs);
		*cluster_num = params.first;
		points = params.second;
	}
	else if (x == 2) {
		int point_num;
		int starting_points;
		std::cout << "Input number of points to be generated." << std::endl;
		std::cin >> point_num;
		std::cout << "Input number of starting point from which others will be derived." << std::endl;
		std::cin >> starting_points;
		std::cout << "Input number of clusters to be created." << std::endl;
		std::cin >> *cluster_num;
		points = generate_random_points(generate_first_points(starting_points), point_num);											// load points

	}
	else {
		std::cout << "Invalid input." << std::endl;
	}

}

void calculate_loads_per_proc(int* points_per, int* displacements, int proc_num, int point_num) {

	int points_to_send = point_num;
	int leftover_points = points_to_send % proc_num;
	int per = points_to_send / proc_num;
	int sum = 0;

	for (int i = 0; i < proc_num; i++) {
		int points_sending = per;
		if (leftover_points) {
			points_sending++;
			leftover_points--;
		}
		points_per[i] = points_sending;
		displacements[i] = sum;
		sum += points_per[i];
	}

}

void create_mpi_point(MPI_Datatype* mpi_point_type) {

	const int    nItems = 4;
	int          blocklengths[nItems] = { 1, 1, 1, 1 };
	MPI_Datatype types[4] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	MPI_Datatype MPI_point_proto;
	MPI_Aint     offsets[nItems];
	std::vector<Point>points;
	points.resize(2);

	offsets[0] = offsetof(Point, master_id);
	offsets[1] = offsetof(Point, x);
	offsets[2] = offsetof(Point, y);
	offsets[3] = offsetof(Point, z);

	MPI_Type_create_struct(nItems, blocklengths, offsets, types, &MPI_point_proto);

	MPI_Aint lb, extent;
	MPI_Type_get_extent(MPI_point_proto, &lb, &extent);

	extent = (char*)&points[1] - (char*)&points[0];

	MPI_Type_create_resized(MPI_point_proto, lb, extent, mpi_point_type);
	MPI_Type_commit(mpi_point_type);


}

void create_mpi_cluster(MPI_Datatype* mpi_cluster_type) {

	const int    nItems = 6;
	int          blocklengths[nItems] = { 1, 1, 1, 1, 1, 1 };
	MPI_Datatype types[nItems] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_SHORT };
	MPI_Datatype MPI_cluster_proto;
	MPI_Aint     offsets[nItems];
	std::vector<Cluster>clusters;
	clusters.resize(2);

	offsets[0] = offsetof(Cluster, id);
	offsets[1] = offsetof(Cluster, x);
	offsets[2] = offsetof(Cluster, y);
	offsets[3] = offsetof(Cluster, z);
	offsets[4] = offsetof(Cluster, WCSS);
	offsets[5] = offsetof(Cluster, changed);

	MPI_Type_create_struct(nItems, blocklengths, offsets, types, &MPI_cluster_proto);

	MPI_Aint lb, extent;
	MPI_Type_get_extent(MPI_cluster_proto, &lb, &extent);

	extent = (char*)&clusters[1] - (char*)&clusters[0];


	MPI_Type_create_resized(MPI_cluster_proto, lb, extent, mpi_cluster_type);
	MPI_Type_commit(mpi_cluster_type);


}


std::vector<Cluster> select_first_centres(std::vector<Point>& points, int number) {

	std::vector<Cluster> clusters;
	clusters.resize(number);
	std::set<int> chosen;
	
	for (int i = 0; i < number; i++) {
		Cluster cluster;
		cluster.x = points[i].x;
		cluster.y = points[i].y;
		cluster.z = points[i].z;
		cluster.id = i;
		points[i].master_id = i;
		clusters[i] = cluster;
	}

	return clusters;

}
int main(int argc, char* argv[]) {

	try {
		int myid = 0;
		int proc_num = 0;
		int cluster_num = 0;
		int point_num = 0;
		int my_points_num = 0;
		double starttime, endtime;
		std::vector<Cluster> clusters_starting;
		std::vector<Point> points_starting;														// all points
		std::vector<Cluster> clusters;
		std::vector<Point> points;																// all points
		std::vector<Point> points_mine;															// points assigned to me

		MPI_Init(&argc, &argv);																    // starts MPI

		MPI_Datatype mpi_point_type, mpi_cluster_type;

		create_mpi_point(&mpi_point_type);														// define types
		create_mpi_cluster(&mpi_cluster_type);													// define types


		MPI_Comm_rank(MPI_COMM_WORLD, &myid);										
		MPI_Comm_size(MPI_COMM_WORLD, &proc_num);									
				

		int* points_per = (int*)malloc(proc_num * sizeof(int));
		int* displacements = (int*)malloc(proc_num * sizeof(int));


		if (myid == 0) {
			start(&cluster_num, points_starting);
			clusters_starting = select_random_centres(points_starting, cluster_num);						// generate first clusters
		}

		std::vector<std::vector<double>> runtimes;
		runtimes.resize(4);

		for (int i = 0; i < 4; i++) {																		
			for (int j = 0; j < 10; j++) {

				clusters.clear();
				points.clear();
				clusters = clusters_starting;
				points = points_starting;

				if (myid == 0) {
					starttime = MPI_Wtime();
					point_num = points.size() / proc_num;
					calculate_loads_per_proc(points_per, displacements, proc_num, points.size());
				}

				omp_set_num_threads(pow(2, i));

				MPI_Bcast(displacements, proc_num, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(points_per, proc_num, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&cluster_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

				clusters.resize(cluster_num);
				my_points_num = points_per[myid];
				points_mine.resize(my_points_num);

				short* cluster_changed = (short*)malloc(cluster_num * sizeof(short));
				for (int i = 0; i < cluster_num; i++) {
					cluster_changed[i] = short(0);
				}

				short changed = 1;
				while (changed) {
					if (myid != 0) {
						clusters.clear();
						clusters.resize(cluster_num);
					}
					MPI_Bcast(clusters.data(), clusters.size(), mpi_cluster_type, 0, MPI_COMM_WORLD);

					MPI_Scatterv(points.data(), points_per, displacements, mpi_point_type, points_mine.data(), my_points_num, mpi_point_type, 0, MPI_COMM_WORLD);

					reassign_points_to_clusts(points_mine, clusters);

					MPI_Gatherv(points_mine.data(), points_mine.size(), mpi_point_type, points.data(), points_per, displacements, mpi_point_type, 0, MPI_COMM_WORLD);

					for (int i = 0; i < cluster_num; i++) {
						MPI_Reduce(&(clusters[i].changed), &cluster_changed[i], 1, MPI_SHORT, MPI_SUM, 0, MPI_COMM_WORLD);
					}

					if (myid == 0) {

						calculate_WCSS(clusters, points);

						changed = clusters_changed(cluster_changed, clusters);

						if (changed == 1) {
							recalculate_centre(points, clusters);
						}

					}

					MPI_Bcast(&changed, 1, MPI_SHORT, 0, MPI_COMM_WORLD);

					if (changed == 0) break;

				}

				endtime = MPI_Wtime();

				if (myid == 0) {
					runtimes[i].push_back(endtime - starttime);
				}

			}

		}

		MPI_Finalize();

		for (int i = 0; i < 4; i++) {
			for (auto runtime : runtimes[i]) {
				std::cout << runtime << " ";
			}
			std::cout << std::endl;
		}
	}
	catch (const char* e) {
		std::cout << e;
		return 1;
	}

	return 0;

}