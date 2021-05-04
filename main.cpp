/* The program computing the effective reflectance for 2-dimension system of shield Coulomb potentials. */

/* Note that you need the GNUPlot.
 *
 * macOS
 *
 * The easiest way to install it thought the Homebrew.
 * If you are not familiar with homebrew, read more about it here: https://brew.sh/
 * To install GNUPlot:
 * brew install gnuplot
 *
 * Linux
 *
 * You know how this works, don't you?
 */

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <tuple>
#include <array>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>

const double Z_H = 1;
const double Z_Al = 13;

const double a_0 = 5.29177210903e-9; //Bohr's radius.

const double Al_distance =  2.86e-10 /  a_0;
const double n = 1.8194e+08 * a_0;
const double alpha_Al = 8.34; //Electronic polarizability.

const double pi = 3.14159265359;

const double e = 1.0;

const std::pair<double, double> throwing_body_size = std::make_pair(1.8897e1, 1.8897e1);

typedef std::pair<double, double> coord;

//The constants below are the minimum and maximum initial particle energies.
int E_min = 20;
int E_max = 200;

int number_of_problems = E_max - (E_min - 1);

std::vector<double> default_energy (number_of_problems); //Just energy range. In main() we will emplace there data.

const double potential_factor = 0.4 * Z_H * Z_Al / (std::pow((std::sqrt(Z_H) + std::sqrt(Z_Al)), 2.0/3.0));

std::vector<std::vector<coord>> trajectory (number_of_problems); //There're points of interactions for every particle.

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

std::vector<coord> crystal_cell (std::pair<double, double> problem_solution_area, double interatomic_distance);

void data_file_creation (std::string DataType, std::vector<coord>& xx);

void crystal_plot ();

std::pair<double, double> default_position_and_direction ();

double random_sighting_parameter_generator (double& b_max);

double distance_from_mass_center (double& Energy) { return std::sqrt(Energy / potential_factor); }

//Read about this function in ReadMe.
double scattering_angle (double& b, double& Energy) { return std::abs(pi - pi*b /
                                                        (std::sqrt(std::pow(b, 2) + potential_factor / Energy)));}

double inelastic_energy_loss (double& b, double& Energy, double& U, double& v);

double Firsovs_shielding (double& Z_min, double& Z_max) { return 0.8853*a_0 *
                                                        std::pow(std::sqrt(Z_min) + std::sqrt(Z_max), -2.0/3.0);}

double velocity (double& Energy, double m) { return std::sqrt(2 * Energy / m); }

coord coordinate_transorm (coord& polar);

void particles_trajectories ();

coord mass_center (coord& particle, coord& potential);

double free_run_length ();

coord offset ();

coord interaction_node (coord& init_point, coord& final_point);

std::vector<coord> subarea (double& x_0, double& c_1, double& c_2, double& dir_cos, double& free_run,
                            std::vector<coord>& nodes);

void general_equation_of_the_line (double& A, double& B, double& C, double x_1, double y_1, double x_2, double y_2);

double distance_from_point_to_line (double& A, double& B, double& C, double& x, double& y);

coord intersection_of_a_line_and_a_circle (coord& center, double& R, coord& inits, double& x_final, double& y_final);

std::vector<double> quadratic_equation_solve (double& a, double& b, double& c);

int main() {
    std::generate(default_energy.begin(), default_energy.end(), [&] {return E_min++;}); //Creating energy range.
    std::vector<coord> nodes = std::move(crystal_cell (throwing_body_size, Al_distance));
    data_file_creation("Nodes", nodes);
    crystal_plot();
    std::vector<coord> default_wave_packets(number_of_problems);
    for (int i = 0; i < number_of_problems; i++)
        default_wave_packets[i] = std::move(default_position_and_direction());
    return 0;
}


template<typename T, size_t... Is>
auto abs_components_impl(T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return (std::sqrt((std::pow(std::get<Is>(t) - std::get<Is>(t1), 2) + ...)));
}

template <class Tuple>
double abs_vector_components (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return abs_components_impl(t, t1,  std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}

coord intersection_of_a_line_and_a_circle (coord& center, double& R, coord& inits, double& x_final, double& y_final) {
    double x_init = inits.first;
    double y_init = inits.second;
    coord intersection;
    double A, B, C;
    double a = center.first;
    double b = center.second;
    general_equation_of_the_line(A, B, C, x_init, y_init, x_final, y_final);
    double alpha = x_init + a;
    double beta = y_init + b;
    double first = std::pow(A, 2) + std::pow(b, 2);
    double second = 2*(alpha*A + beta*B);
    double third = std::pow(alpha, 2) + std::pow(beta, 2);
    std::vector<double> t = std::move(quadratic_equation_solve(first, second, third));
    double distance = 1.0e300;
    for (int i = 0; i < t.size(); i++) {
        double x = x_init + A*t[i];
        double y = y_init + B*t[i];
        coord solution = std::make_pair(x, y);
        intersection = (abs_vector_components(solution, inits) < distance) ? solution : intersection;
    }
    return intersection;
}

//Function returns roots of quadratic equation (a*t^2 + b^2*t + c == 0) taking coefficients.
std::vector<double> quadratic_equation_solve (double& a, double& b, double& c) {
    double D = std::pow(b, 2) - 4*a*c;
    double t_1, t_2;
    if (D >= 0 && !std::isnan(D)) {
        t_1 = (-b + std::sqrt(sqrt(D))) / 2.0 / a;
        t_2 = (-b - std::sqrt(sqrt(D))) / 2.0 / a;
    } else
        t_1 = t_2 = 0;
    return {t_1, t_2};
}


//First of all lets generate the direction of particle.
double direction_cos () { //returns cos angle.
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return 2*dis(gen) - 1;
}



//Then we have to know the velocity of particle.



//Then we have to define the interaction node
coord interaction_node (coord& init_point, double& free_run, double& dir_cos, double& b_max,
                        std::vector<coord>& nodes) {
    double A, B, C, r, d;
    int i = 0;
    double x_1 = init_point.first;
    double y_1 = init_point.second;
    double x_2 = free_run*dir_cos + x_1;
    double y_2 = free_run*std::sin(std::acos(dir_cos)) + y_1;
    std::vector<coord> subnodes = std::move(subarea(x_1, y_1, x_2, y_2, b_max, nodes));
    general_equation_of_the_line(A, B, C, x_1, y_1, x_2, y_2);
    do {
        r = random_sighting_parameter_generator(b_max);
        d = distance_from_point_to_line(A, B, C, subnodes[i].first, subnodes[i].second);
        i++;
    } while (r <= d && i < subnodes.size());
    return subnodes[i];
}

//Function determines nodes, which could make an influence on particle.
std::vector<coord> subarea (double& x_1, double& y_1, double& x_2, double& y_2, double& b_max,
                            std::vector<coord>& nodes) {
    std::vector<coord> subnodes;
    double k = (y_2 - y_1)/(x_2 - x_1);
    double x_min = std::min(x_1, x_2);
    double x_max = std::min(x_1, x_2);
    for (int i = 0; i < nodes.size(); i++) {
        double x = nodes[i].first;
        double y = nodes[i].second;
        double y_up = k*x + y_1+b_max;
        double y_down = k*x + y_1-b_max;
        if (x >= x_min && x <= x_max && y <= y_up && y >= y_down)
            subnodes.emplace_back(nodes[i]);
    }
    return subnodes;
}

void general_equation_of_the_line (double& A, double& B, double& C, double x_1, double y_1, double x_2, double y_2) {
    A = y_1 - y_2;
    B = x_2 - x_1;
    C = x_1*y_2 - x_2*y_1;
}

double distance_from_point_to_line (double& A, double& B, double& C, double& x, double& y) {
    //d = |A*x + B*y + C| / sqrt(A^2 + B^2)
    return std::abs(A*x + B*y + C) / std::sqrt(std::pow(A, 2) + std::pow(B, 2));
}

double free_run_length (double& v, double alpha, double mu) {
    double sigma_polar = 2 * pi * e / v * std::sqrt(alpha / mu);
    double lambda_average = 1.0 / n * sigma_polar;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return -lambda_average * std::log(dis(gen));
}


/*coord mass_center (coord& particle, coord& potential) {

}*/

void particle_trajectories (std::vector<double>& E_init) {
    for (int i = 0; i < E_init.size(); i++) {
        double E_0 = 10;
        double E = E_init[i];
        double x, y, b, U, v;
        do {
            //while (/*b-statment*/) {}

            trajectory[i].emplace_back(std::move(std::make_pair(x, y)));
            E -= inelastic_energy_loss(b, E, U, v);
        } while (E > E_0 && x > 0);
    }
}

//Function takes as input coordinates in polar system and return the coordinates in descart coordinate system.
coord coordinate_transorm (coord& polar) {
    double phi = polar.first;
    double rho = polar.second;
    double x = rho * cos(phi);
    double y = rho * sin(phi);
    return std::move(std::make_pair(x, y));
}

//Think about this. Particles lose there's Energy anyway!
double inelastic_energy_loss (double& b, double& Energy, double& U, double& v) {
    double Z_min = std::min(Z_Al, Z_H);
    double Z_max = std::max(Z_Al, Z_H);
    double r_min = std::sqrt((Energy * std::pow(b, 2) + potential_factor) / Energy); // r_min = r_min(b)
    return 0.3 * Z_Al * (std::sqrt(Z_max) + std::sqrt(Z_min)) * (std::pow(Z_max, 1.0/6.0) + std::pow(Z_min, 1.0/6.0)) /
            (1+0.67*std::sqrt(Z_max)*r_min /
            std::pow(Firsovs_shielding(Z_min, Z_max) * (std::pow(Z_max, 1.0/6.0) + std::pow(Z_min, 1.0/6.0)), 3.0)) *
                       (1 - 0.68 * U/Energy) * v;
}

//Function returns the sighting parameter for scattering angle determination.
double random_sighting_parameter_generator (double& b_max) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double gamma = dis(gen);
    return b_max*std::sqrt(gamma);
}

std::pair<double, double> default_position_and_direction () {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double phi = pi * dis(gen) - pi/2;
    double y = throwing_body_size.second * dis(gen);
    return std::move(std::make_pair(phi, y));
}

//It will be updated later.
void crystal_plot () {
    //if you have problems with ".svg", you have to change ".svg" to ".pdf" in strings bellow.
    FILE *gp = popen("gnuplot  -persist", "w");
    if (!gp)
        throw std::runtime_error("Error opening pipe to GNU plot.");
    else {
        std::vector<std::string> stuff = {"set term svg",
                                          "set out \'Nodes.svg\'",
                                          "set grid xtics ytics",
                                          "set title \'Cristal cell\'",
                                          "set xlabel \'x, Bohr radius\'",
                                          "set ylabel \'y, Bohr radius\'",
                                          "set xrange [0:1]",
                                          "set yrange [0:1]",
                                          "plot \'Nodes\' using 1:2 lw 1 lt rgb 'orange' ti \'Nodes\'",
                                          "set key off",
                                          "set terminal wxt",
                                          "set output",
                                          "replot", "q"};
        for (const auto &it : stuff)
            fprintf(gp, "%s\n", it.c_str());
        pclose(gp);
    }
}

std::vector<coord> crystal_cell (std::pair<double, double> problem_solution_area, double interatomic_distance) {
    double width = problem_solution_area.first;
    double length = problem_solution_area.second;
    double w, l;
    int i = 0;
    w = l = 0;
    std::vector<coord> nodes;
    while (w < width) {
        while (l <= length) {
            nodes.emplace_back(std::move(std::make_pair(l, w)));
            l += interatomic_distance / sin(pi/4);
        }
        l = (i % 2 == 0) ? interatomic_distance / sin(pi/4) / 2.0 : 0;
        i++;
        w += interatomic_distance * sin(pi/4);
    }
    return nodes;
}

void data_file_creation (std::string DataType, std::vector<coord>& xx) {
    //For reading created files via Matlab use command: M = dlmread('/PATH/file'); xi = M(:,i);
    std::ofstream fout;
    fout.open(DataType);
    for (int i = 0; i < xx.size(); i++)
        fout << xx[i].first << '\t' << xx[i].second << std::endl;
    fout.close();
}