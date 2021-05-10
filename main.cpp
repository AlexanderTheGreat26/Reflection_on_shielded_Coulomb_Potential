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
#include <stdexcept>


const double m_e = 5.48579909065e-4; // (atomic mass unit)
const double v_e = 2.2e8; // (atomic velocity unit)
const double Z_H = 1;
const double m_H = 1.00811 / m_e; // (electron mass)
const double Z_Al = 13;
const double m_Al = 26.9815386 / m_e; // (electron mass)
const double mu = m_H*m_Al / (m_H+m_Al); //reduced mass.
const double E_h = 27.2113845;

const double a_0 = 5.29177210903e-9; //Bohr's radius (cm)

const double Al_interatomic_distance =  2.86e-8 / a_0; // Bohr's radius
//const double n = 3.3102e16 * std::pow(a_0, 2); // 1/(Bohr radius)^2
const double n = 6.0226e+24 * std::pow(a_0, 3);
const double alpha_Al = 8.34; //Electronic polarizability.

const double pi = 3.14159265359;

const double e = 1.0;

const double E_final = 10 / E_h;

const std::pair<double, double> throwing_body_size = std::make_pair(2.0e4, 2.0e4); //(bohr radius)

std::vector<std::tuple<double, double, double>> throwing_body_size_borders;

typedef std::pair<double, double> coord;


//The constants below are the minimum and maximum initial particle energies.
int E_min = 20;
int E_max = 30;

int number_of_problems = E_max - (E_min - 1);

std::vector<double> default_energy (number_of_problems); //Just energy range. In main() we will emplace there data.

const double potential_factor = 0.4 * Z_H * Z_Al / (std::pow((std::sqrt(Z_H) + std::sqrt(Z_Al)), 2.0/3.0));

std::vector<std::pair<int, double>> outside; //the array of number of problem and final energy

std::vector<double> momentum (number_of_problems);

std::vector<std::vector<coord>> trajectory (number_of_problems); //There're points of interactions for every particle.

std::vector<std::vector<std::pair<coord, double>>> sighting_param (number_of_problems); //There's we will store the coordinates of nodes and b;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

std::vector<coord> crystal_cell (std::pair<double, double> problem_solution_area, double interatomic_distance);

void data_file_creation (std::string DataType, std::vector<coord>& xx);

void crystal_plot ();

std::pair<double, double> default_position_and_direction ();

double random_sighting_parameter_generator (double& b_max, double& b_min);

double distance_from_mass_center (double& Energy) { return std::sqrt(potential_factor / Energy); }

//Read about this function in ReadMe.
double scattering_angle (double& b, double& Energy) { return std::abs(pi - pi*b /
                                                        (std::sqrt(std::pow(b, 2) + potential_factor / Energy)));}

double inelastic_energy_loss (double& r, double& Energy, double& U, double v);

double elastic_energy_loss (double& dir_cos, double& v_0);

double Firsovs_shielding (double& Z_min, double& Z_max) { return 0.8853*a_0 *
                                                        std::pow(std::sqrt(Z_min) + std::sqrt(Z_max), -2.0/3.0);}

double velocity (double& Energy, double m) { return std::sqrt(2 * Energy / m); } // (v_e)

coord coordinate_transorm (coord& polar);

void particles_trajectories ();

double mass_center (double particle_mass, double potential_mass, double& b);

double free_run_length ();

coord interaction_node (coord& init_point, coord& final_point, double& b_max,
                        std::vector<coord>& nodes, double& closest_approach_radius, double& b);

std::vector<coord> subarea (double& x_1, double& y_1, double& x_2, double& y_2, double& b_max,
                            std::vector<coord>& nodes);

void general_equation_of_the_line (double& A, double& B, double& C, double x_1, double y_1, double x_2, double y_2);

double distance_from_point_to_line (double& A, double& B, double& C, double& x, double& y);

coord intersection_of_a_line_and_a_circle (coord& center, double& R, coord& inits, coord maximum_trajectory);

std::vector<double> quadratic_equation_solve (double& a, double& b, double& c);

double direction_cos ();

void particle_wander (std::vector<double>& Energy, coord& initial_coordinate, std::vector<coord>& nodes);

coord vector_offset (coord& frame_of_reference, coord& vector);

coord vector_creation (coord& end, coord& begin);

coord rotation_matrix (coord& vector, double& theta);

void coordinate_shift (std::vector<coord>& points, double dx, double dy);

double cos_t (coord& a, coord& b);

double shielded_Coulomb_potential (double& r) { return potential_factor / std::pow(r, 2); }

void coordinates_from_pair (double& x, double& y, coord& data) { x = data.first; y = data.second; }

void area_borders_creation (std::vector<std::tuple<double, double, double>>& borders, std::vector<coord>& nodes);


int main() {
    std::generate(default_energy.begin(), default_energy.end(), [&] { return ++E_min/E_h; }); //Creating energy range.
    std::vector<coord> nodes = std::move(crystal_cell (throwing_body_size, Al_interatomic_distance));
    coordinate_shift(nodes, throwing_body_size.first/2.0, throwing_body_size.second/2.0);
    area_borders_creation(throwing_body_size_borders, nodes);

    data_file_creation("Nodes", nodes);
    coord init = std::make_pair(-throwing_body_size.first/2.0, 0);
    particle_wander(default_energy, init, nodes);

    crystal_plot();
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

template<typename T, size_t... Is>
auto scalar_prod_components_impl(T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return ((std::get<Is>(t) * std::get<Is>(t1)) + ...);
}

template<typename T, size_t... Is>
auto abs_components_impl(T const& t, std::index_sequence<Is...>) {
    return std::sqrt((std::pow(std::get<Is>(t), 2) + ...));
}

template <class Tuple>
double abs_components(const Tuple& t) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return abs_components_impl(t, std::make_index_sequence<size>{});
}

template <class Tuple>
double scalar_prod_components(const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return scalar_prod_components_impl(t, t1,  std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}

double cos_t (coord& a, coord& b) {
    return scalar_prod_components(a, b) / (abs_components(a) * abs_components(b));
}

void coordinate_shift (std::vector<coord>& points, double dx, double dy) {
    for (int i = 0; i < points.size(); ++i) {
        points.at(i).first -= dx;
        points.at(i).second -= dy;
    }
}

void area_borders_creation (std::vector<std::tuple<double, double, double>>& borders, std::vector<coord>& nodes) {
    // We set the global variable of throwing body size. So we will create the borders of task area by
    // general line equation (Ax + By + C = 0).
    double a = nodes[0].first;
    double b = nodes[nodes.size()-1].second;
    std::vector<coord> area_vertices = {std::make_pair(-a, -b),
                                        std::make_pair(-a, b),
                                        std::make_pair(a, b),
                                        std::make_pair(a, -b)};
    double A, B, C;
    general_equation_of_the_line(A, B, C, area_vertices[0].first, area_vertices[0].second,
                                 area_vertices[area_vertices.size()-1].first, area_vertices[area_vertices.size()-1].second);
    borders.emplace_back(std::move(std::make_tuple(A, B, C)));
    for (int i = 1; i < area_vertices.size(); ++i) {
        general_equation_of_the_line(A, B, C, area_vertices[i-1].first, area_vertices[i-1].second,
                                              area_vertices[i].first, area_vertices[i].second);
        borders.emplace_back(std::move(std::make_tuple(A, B, C)));
    }
}

//Well, I shame about this, but LU-Decomposition doesn't works and i have no time to fix it. May be I'll rewrite this later.

double det (std::vector<std::array<double, 2>>& matrix) {
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
}

std::vector<std::array<double, 2>> replace (std::vector<std::array<double, 2>> matrix, std::vector<double>& b, int& k) {
    for (int i = 0; i < 2; ++i)
        matrix[i][k] = b[i];
    return matrix;
}

coord solve (std::vector<std::array<double, 2>> matrix, std::vector<double>& free_numbers_column) {
    double D = det(matrix);
    std::vector<double> dets;
    for (int k = 0; k < 2; ++k) {
        std::vector<std::array<double, 2>> M = replace(matrix, free_numbers_column, k);
        dets.emplace_back(det(M) / D);
    }
    return std::move(std::make_pair(dets[0], dets[1]));
}

bool directions_coincidence (coord& a, coord& b) { return (a.first * b.first > 0 && a.second * b.second > 0); }

coord border_intersection (std::vector<std::tuple<double, double, double>>& borders, coord& inits, coord& finals) {
    double A, B, C;
    coord direction = vector_creation(finals, inits);
    general_equation_of_the_line(A, B, C, inits.first, inits.second, finals.first, finals.second);
    coord intersection;
    for (int i = 0; i < borders.size(); ++i) {
        double A_border = std::get<0>(borders[i]);
        double B_border = std::get<1>(borders[i]);
        double C_border = std::get<2>(borders[i]);
        std::vector<std::array<double, 2>> matrix = {{A_border, B_border},
                                                     {A, B}};
        std::vector<double> right_part = {-C_border, -C};
        finals = std::move(solve(matrix, right_part));
        coord test_direction = vector_creation(finals, inits);
        if (directions_coincidence(direction, test_direction) && abs_components(test_direction) > 1.0e-15
            && std::abs(finals.first) <= throwing_body_size.first / 2.0
            && std::abs(finals.second) <= throwing_body_size.second / 2.0) {
            intersection = std::move(finals);
            break;
        }
    }
    return intersection;
}

void particle_wander (std::vector<double>& Energy, coord& initial_coordinate, std::vector<coord>& nodes) {
    for (int i = 0; i < number_of_problems; ++i) {
        double E = Energy[i];
        double dir_cos = direction_cos();
        double x_1, y_1, x_2, y_2, b, A, B, C;
        coord particle_coordinate = initial_coordinate;
        x_1 = particle_coordinate.first;
        y_1 = particle_coordinate.second;
        double v = velocity(E, m_H);
        double p = m_H * v * dir_cos;
        do {
            double r = distance_from_mass_center(E);
            double b_max = Al_interatomic_distance / 2.0;;

            double l = free_run_length();
            x_2 = l*dir_cos + x_1;
            y_2 = l*std::sin(std::acos(dir_cos)) + y_1;
            coord direction = std::make_pair(x_2, y_2);
            coord free_run = std::move(border_intersection(throwing_body_size_borders, particle_coordinate,
                                                           direction));


            // Attention! We get the parameter b gets by reference from function interaction_node(...)!
            // We don't need to play it anywhere more!

            coord scattering_potential = interaction_node(particle_coordinate, free_run, b_max,
                                                          nodes, r, b);
            if (scattering_potential.first > throwing_body_size.first) {
                trajectory[i].emplace_back(free_run);
                break;
            }
            std::cout << "scat:\t" << scattering_potential.first << '\t' << scattering_potential.second << std::endl;

            particle_coordinate = intersection_of_a_line_and_a_circle(scattering_potential, b,
                                                                      particle_coordinate, free_run);
            double theta = scattering_angle(b, E);
            coord rho = vector_creation(particle_coordinate, scattering_potential);
            coord rotation = rotation_matrix(rho, theta);
            coord new_direction = vector_offset(particle_coordinate, rotation);
            dir_cos = cos_t(free_run, new_direction);
            if (E >= 100/E_h) {
                double U = shielded_Coulomb_potential(r);
                E -= (inelastic_energy_loss(r, E, U, v) / E_h);
                v = velocity(E, m_H);
            } else {
                E -= elastic_energy_loss(dir_cos, v);
                //Emplace code of momentum analyze here.
            }
            x_1 = particle_coordinate.first;
            y_1 = particle_coordinate.second;
            trajectory[i].emplace_back(particle_coordinate);
            sighting_param[i].emplace_back(std::make_pair(scattering_potential, b));
            if (x_1 <= -throwing_body_size.first/2.0) {
                outside.emplace_back(std::make_pair(i, E));
                coord abscissa_axis_dir = std::make_pair(1.0, 0.0);
                momentum[i] = p - m_H * v * cos_t(free_run, abscissa_axis_dir);
                break;
            }
        } while (E > E_final && std::abs(x_1) < throwing_body_size.first/2.0
                             && std::abs(y_1) < throwing_body_size.second/2.0);
    }
}

coord rotation_matrix (coord& vector, double& theta) {
    double x = vector.first;
    double y = vector.second;
    double x_theta = x*std::cos(theta) - y*std::sin(theta);
    double y_theta = x*std::sin(theta) + y*std::cos(theta);
    return std::make_pair(x_theta, y_theta);
}

coord vector_creation (coord& end, coord& begin) { return std::move(std::make_pair(end.first - begin.first,
                                                                                   end.second - begin.second)); };

coord vector_offset (coord& frame_of_reference, coord& vector) {
    return std::make_pair(frame_of_reference.first + vector.first, frame_of_reference.second + vector.second);
}

coord intersection_of_a_line_and_a_circle (coord& center, double& R, coord& inits, coord maximum_trajectory) {
    double x_init = inits.first;
    double y_init = inits.second;
    double x_final = maximum_trajectory.first;
    double y_final = maximum_trajectory.second;
    coord intersection;
    double A, B, C;
    double a = center.first;
    double b = center.second;
    general_equation_of_the_line(A, B, C, x_init, y_init, x_final, y_final);
    double alpha = x_init - a;
    double beta = y_init - b;
    double first = std::pow(A, 2) + std::pow(B, 2);
    double second = 2*(B*alpha - A*beta);
    double third = std::pow(alpha, 2) + std::pow(beta, 2) - std::pow(R, 2);
    std::vector<double> t = std::move(quadratic_equation_solve(first, second, third));
    double x, y, distance = 1.0e300;
    for (int i = 0; i < t.size(); ++i) {
        x = B*t[i] + x_init;
        y = -A*t[i] + y_init;
        coord solution = std::make_pair(x, y);
        intersection = (abs_vector_components(solution, inits) <= distance) ? solution : intersection;
        distance = abs_vector_components(solution, inits);
    }
    return intersection;
}

//Function returns roots of quadratic equation (a*t^2 + b^2*t + c == 0) taking coefficients.
std::vector<double> quadratic_equation_solve (double& a, double& b, double& c) {
    double D = std::pow(b, 2) - 4*a*c;
    double t_1, t_2;
    if (D >= 0 && !std::isnan(D)) {
        t_1 = (-b + std::sqrt(D)) / 2.0 / a;
        t_2 = (-b - std::sqrt(D)) / 2.0 / a;
    } else {
        std::cout << "D:\t" << D << std::endl;
        t_1 = t_2 = 0;
    }
    return {t_1, t_2};
}


//First of all lets generate the direction of particle.
double direction_cos () { //returns cos angle.
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

//Then we have to define the interaction node and closest approach radius.
coord interaction_node (coord& init_point, coord& final_point, double& b_max,
                        std::vector<coord>& nodes, double& closest_approach_radius, double& b) {
    double A, B, C, d;
    int i = 0;

    double x_1 = init_point.first;
    double y_1 = init_point.second;

    double x_2 = final_point.first;
    double y_2 = final_point.second;

    std::vector<coord> subnodes = std::move(subarea(x_1, y_1, x_2, y_2, b_max, nodes));
    if (subnodes.empty()) return std::make_pair(1.0e308, 1.0e308);

    //test other parts before using it!
    //std::sort(subnodes.begin(), subnodes.end(), [&] { return abs_vector_components(init_point, subnodes[i]);} );

    general_equation_of_the_line(A, B, C, x_1, y_1, x_2, y_2);
    do {
        b = random_sighting_parameter_generator(b_max, closest_approach_radius);
        d = distance_from_point_to_line(A, B, C, subnodes[i].first, subnodes[i].second);
        ++i;
    } while (b < d && i < subnodes.size());
    if (b >= d)
        return subnodes[i-1];
    else {
        //b = b_max; //May be exclude this case and say that particle had no interactions.
        //return subnodes[subnodes.size()-1];
        return std::make_pair(1.0e308, 1.0e308);
    }
}

//Function determines nodes, which could make an influence on particle.
std::vector<coord> subarea (double& x_1, double& y_1, double& x_2, double& y_2, double& b_max,
                            std::vector<coord>& nodes) {
    std::vector<coord> subnodes;
    double x_min = std::min(x_1, x_2);
    double x_max = std::max(x_1, x_2);
    double k = (y_2 - y_1)/(x_max - x_min);
    double b = y_1 - k*x_1;
    for (int i = 0; i < nodes.size(); ++i) {
        double x = nodes[i].first;
        double y = nodes[i].second;
        double y_up = k*x + b+b_max/2.0;
        double y_down = k*x + b-b_max/2.0;
        if (x >= x_min && x <= x_max && y >= y_down && y <= y_up)
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

double free_run_length () {

    // Well, it's crunch, so I'll rewrite it when the task allows to estimate the scattering cross section
    // by another method or I have enough computing resources for large area. Now the function returns maximum length
    // could be included into the task area.

    return throwing_body_size.first * std::sqrt(2.0);
}

double inelastic_energy_loss (double& r, double& Energy, double& U, double v) {
    double Z_min = std::min(Z_Al, Z_H);
    double Z_max = std::max(Z_Al, Z_H);
    double r_min = r * a_0;
    v *= v_e;
    return 0.3e-7 * Z_Al * (std::sqrt(Z_max) + std::sqrt(Z_min)) * (std::pow(Z_max, 1.0/6.0) + std::pow(Z_min, 1.0/6.0)) /
            (1 + 0.67 * std::sqrt(Z_max)*r_min /
            std::pow(Firsovs_shielding(Z_min, Z_max) * (std::pow(Z_max, 1.0/6.0) + std::pow(Z_min, 1.0/6.0)), 3.0)) *
                       (1 - 0.68 * U/Energy) * v;
}

double kinetic_energy (double m, double& v) {
    return m*std::pow(v, 2) / 2.0;
}

double elastic_energy_loss (double& dir_cos, double& v_0) {
    double a = 1.0;
    double b = -2* mu/m_Al * dir_cos;
    double c = -(m_Al - m_H) / (m_Al + m_H);
    std::vector<double> t = std::move(quadratic_equation_solve(a, b, c));
    double velocity_ratio = (t[0] > 0) ? t[0] : t[1];
    double E_init = kinetic_energy(m_H, v_0);
    v_0 *= velocity_ratio; // v / v_0
    double E_fin = kinetic_energy(m_H, v_0);
    return E_init - E_fin;
}

//Function returns the sighting parameter for scattering angle determination.
double random_sighting_parameter_generator (double& b_max, double& b_min) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double c = std::pow(b_max, 2) - std::pow(b_min, 2);
    double gamma = dis(gen);
    return std::sqrt(gamma + std::pow(b_min, 2) / c) * c;
}

std::pair<double, double> default_position_and_direction () {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double phi = pi * dis(gen) - pi/2;
    double y = throwing_body_size.second * dis(gen);
    return std::move(std::make_pair(phi, y));
}

//It will be updated later.
void crystal_plot () {
    double x, y, b;
    FILE *gp = popen("gnuplot  -persist", "w");
    if (!gp)
        throw std::runtime_error("Error opening pipe to GNU plot.");
    else {
        std::vector<std::string> stuff = {"set term pop",
                                          "set multiplot",
                                          "set grid xtics ytics",
                                          "set title \'Cristal cell\'",
                                          "set xlabel \'x, Bohr radius\'",
                                          "set ylabel \'y, Bohr radius\'",
                                          "set xrange [-10000:-9900]",
                                          "set yrange [-500:500]",
                                          "plot \'Nodes\' using 1:2 lw 1 lt rgb 'orange' ti \'Nodes\'",
                                          "set key off",
                                          "plot \'Nodes\'",
                                          "plot '-' u 1:2 w lines"};
        for (const auto &it : stuff)
            fprintf(gp, "%s\n", it.c_str());
        for (int i = 0; i < number_of_problems; ++i) {
            for (int j = 0; j < trajectory[i].size(); ++j) {
                coordinates_from_pair(x, y, trajectory[i][j]);
                fprintf(gp, "%f\t%f\n", x, y);
            }
            fprintf(gp, "%c\n%s\n", 'e', "plot '-' u 1:2 w lines");
        }
        for (int i = 0; i < number_of_problems; ++i) {
            fprintf(gp, "%s\n", (i < number_of_problems-1) ? "$Data <<EOD" : "q");
            for (int j = 0; j < sighting_param[i].size(); ++j) {
                coordinates_from_pair(x, y, sighting_param[i][j].first);
                b = sighting_param[i][j].second / 100.0 / 1000.0; //
                fprintf(gp, "%f\t%f\t%f\n", x, y, b);
            }
            fprintf(gp, "%s\n%s\n", "EOD", (i < number_of_problems-1) ?
            "plot $Data u 1:2:3 w p ps var pt 6 lc 'black'" : "q");
        }
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
        w += interatomic_distance * sin(pi/4);
        ++i;
    }
    return nodes;
}

void data_file_creation (std::string DataType, std::vector<coord>& xx) {
    //For reading created files via Matlab use command: M = dlmread('/PATH/file'); x_i = M(:,i);
    std::ofstream fout;
    fout.open(DataType);
    for (int i = 0; i < xx.size(); ++i)
        fout << xx[i].first << '\t' << xx[i].second << std::endl;
    fout.close();
}
