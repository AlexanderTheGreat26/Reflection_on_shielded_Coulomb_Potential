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

const std::pair<double, double> throwing_body_size = std::make_pair(1.8897e1, 1.8897e1);

typedef std::pair<double, double> coord;

//The constants below are the minimum and maximum initial particle energies.
int E_min = 20;
int E_max = 200;

int number_of_problems = E_max - (E_min - 1);

std::vector<double> E (number_of_problems); //Just energy range. In main() we will emplace there.

const double potential_factor = 0.4 * Z_H * Z_Al / (std::pow((std::sqrt(Z_H) + std::sqrt(Z_Al)), 2.0/3.0));

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

std::vector<coord> crystal_cell (std::pair<double, double> problem_solution_area, double interatomic_distance);

void data_file_creation (std::string DataType, std::vector<coord>& xx);

std::vector<double> distance_from_mass_center (std::vector<double>& E);

void crystal_plot ();

std::pair<double, double> default_position_and_direction ();

int main() {
    std::generate(E.begin(), E.end(), [&] {return E_min++;}); //Creating energy range.
    std::vector<double> r = std::move(distance_from_mass_center(E));
    std::vector<coord> nodes = std::move(crystal_cell (throwing_body_size, Al_distance));
    data_file_creation("Nodes", nodes);
    crystal_plot();
    std::vector<coord> default_wave_packets(number_of_problems);
    for (int i = 0; i < number_of_problems; i++)
        default_wave_packets[i] = (std::move(default_position_and_direction()));
    return 0;
}

std::pair<double, double> default_position_and_direction () {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double phi = pi * dis(gen) - pi/2;
    double y = throwing_body_size.second * dis(gen);
    return std::move(std::make_pair(phi, y));
}

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

std::vector<double> distance_from_mass_center (std::vector<double>& E) {
    std::vector<double> r(number_of_problems);
    for (int i = 0; i < number_of_problems; i++)
        r[i] = std::sqrt(E[i] / potential_factor);
    return r;
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