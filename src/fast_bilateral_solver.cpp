/* Copyright 2017 Toon Van den Zegel. All Rights Reserved.								*/
/*																						*/
/* This file is part of fast_bilateral_space_stereo.									*/
/*																						*/
/* fast_bilateral_space_stereo is free software :										*/
/* you can redistribute it and / or modify												*/
/* it under the terms of the GNU General Public License as published by					*/
/* the Free Software Foundation, either version 3 of the License, or					*/
/* (at your option) any later version.													*/
/*																						*/
/* fast_bilateral_space_stereo is distributed in the hope that it will be useful,		*/
/* but WITHOUT ANY WARRANTY; without even the implied warranty of						*/
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the							*/
/* GNU General Public License for more details.											*/
/*																						*/
/* You should have received a copy of the GNU General Public License					*/
/* along with fast_bilateral_space_stereo.												*/
/* If not, see <http://www.gnu.org/licenses/>.											*/

#include "fast_bilateral_solver.h"

fast_bilateral_solver::fast_bilateral_problem::fast_bilateral_problem(
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat_C_CBC,
	const bilateral_grid_simplified& grid,
	const std::vector<int>& lookup,
	int disparty_range,
	float lambda,
	int keep_nb_of_intermediate_images) :
	mat_C_CBC(mat_C_CBC),
	grid(grid),
	lookup(lookup),
	disparty_range(disparty_range),
	lambda(lambda),
	nb_vertices(grid.get_nb_vertices()),
	keep_nb_of_intermediate_images(keep_nb_of_intermediate_images)
{
	intermediate_prev_energy = std::numeric_limits<double>::max();
}

bool fast_bilateral_solver::fast_bilateral_problem::Evaluate(const double* const parameters,
	double* cost,
	double* gradient) const 
{
	// the solver works with doubles, but our functions uses float, not very efficient :/

	Eigen::VectorXf x_eig = Eigen::VectorXf(nb_vertices);
	Eigen::VectorXf grad_eig = Eigen::VectorXf(nb_vertices);

	// convert to an eigen vector (not very efficient in this way)
	for (int i = 0; i < nb_vertices; ++i)
	{
		x_eig(i) = static_cast<float>(parameters[i]);
	}

	/// Smoothing term!

	// smoothing energy loss
	float loss_smoothing_term = x_eig.transpose() *	mat_C_CBC * x_eig;

	// smoothing gradient
	grad_eig = (mat_C_CBC * x_eig) * 2.0f;
	for (int i = 0; i < nb_vertices; ++i)
	{
		gradient[i] = grad_eig(i);
	}

	/// Data term
	const float fdisparty_max = (float)disparty_range;
	float loss_data_term = 0.f;
	for (int j = 0; j < nb_vertices; ++j)
	{
		const float vj = static_cast<float>(parameters[j]);

		float grad_data = 0.f;
		// is the disparity value in range of the lookup table?
		if (vj < 0.f)
		{
			// no
			grad_data = -(border_cost_value - 0.f);
			loss_data_term += 500.0f;
		}
		else if (vj >= fdisparty_max)
		{
			// no
			grad_data = -(0.f - border_cost_value);
			loss_data_term += 500.0f;
		}
		else
		{
			// yes!

			// a linear interpolation between lookup values
			const int vj_id = static_cast<int>(std::floor(vj));
			const int lookup_idx = j * disparty_range + vj_id;
			const float gj_floor = static_cast<float>(lookup[lookup_idx]);
			const float gj_ceil = static_cast<float>(lookup[lookup_idx + 1]);

			grad_data = -(gj_floor - gj_ceil);

			const float v1 = ((float)(vj_id + 1) - vj) * gj_floor;
			const float v2 = (vj - (float)vj_id) * gj_ceil;
			loss_data_term += v1 + v2;
		}

		gradient[j] += grad_data *lambda;
	}

	// final energy term (combine data and smoothing term)
	float loss_total = loss_smoothing_term + lambda * loss_data_term;
	cost[0] = loss_total;

	// keep result of each iteration if needed
	if ((loss_total < intermediate_prev_energy) && (intermediate_results.size() < keep_nb_of_intermediate_images))
	{
		intermediate_results.push_back(x_eig);
		intermediate_prev_energy = loss_total;
	}


	return true;
}

fast_bilateral_solver::fast_bilateral_solver()
{
	//google::InitGoogleLogging(nullptr);
}

void fast_bilateral_solver::bistochastize(const bilateral_grid_simplified& grid, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dn, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dm, const int nb_iterations)
{
	Eigen::MatrixXf mat_pixs(grid.get_splat_matrix().innerSize(), 1);
	mat_pixs.setOnes();
	Eigen::MatrixXf mat_m = grid.get_splat_matrix() * mat_pixs;

	Eigen::MatrixXf mat_n(grid.get_nb_vertices(), 1);
	mat_n.setOnes();

	// bistochastize method
	Eigen::MatrixXf mat_n_new;
	for (int i = 0; i < nb_iterations; ++i)
	{

		Eigen::MatrixXf ress = (mat_n.cwiseProduct(mat_m)).cwiseQuotient(grid.blur(mat_n));

		mat_n_new = ress.cwiseSqrt();

		Eigen::MatrixXf diff = mat_n_new - mat_n;
		float fdelta = diff.sum();
		std::cout << "bistochastize delta: " <<  fdelta << std::endl;
		mat_n = mat_n_new;
	}
	mat_m = mat_n.cwiseProduct(grid.blur(mat_n));

	// Convert result to diagonal matrices (could not find correct function in eigen :/)
	Dm = Eigen::SparseMatrix<float, Eigen::RowMajor>(grid.get_nb_vertices(), grid.get_nb_vertices());
	{
		typedef Eigen::Triplet<float> T;
		std::vector<T> tripletList;
		for (int i = 0; i < grid.get_nb_vertices(); ++i)
		{
			tripletList.push_back(T(i, i, mat_m(i, 0)));

		}
		Dm.setFromTriplets(tripletList.begin(), tripletList.end());
	}

	Dn = Eigen::SparseMatrix<float, Eigen::RowMajor>(grid.get_nb_vertices(), grid.get_nb_vertices());
	{
		typedef Eigen::Triplet<float> T;
		std::vector<T> tripletList;
		for (int i = 0; i < grid.get_nb_vertices(); ++i)
		{
			tripletList.push_back(T(i, i, mat_n.coeffRef(i, 0)));

		}
		Dn.setFromTriplets(tripletList.begin(), tripletList.end());
	}
}

void fast_bilateral_solver::splat_initial_image(const bilateral_grid_simplified& grid, const cv::Mat initial_image_float, double* parameters)
{
	assert(initial_image_float.type() == CV_32FC1);
	const int nb_pixels = initial_image_float.cols * initial_image_float.rows;
	Eigen::VectorXf in_vec(nb_pixels);
	const float *pff = reinterpret_cast<const float*>(initial_image_float.data);
	for (int i = 0; i < nb_pixels; ++i)
	{
		in_vec(i) = pff[i];
	}

	// splat input image
	Eigen::VectorXf splat_in_vec = grid.get_splat_matrix() * in_vec;

	// the grid is not normalized, so splat a bunch of 'ones' so we can normalize the result
	Eigen::MatrixXf normalization_vec(nb_pixels, 1);
	normalization_vec.setOnes();
	Eigen::MatrixXf normalization_weights = grid.get_splat_matrix() * normalization_vec;
	splat_in_vec = splat_in_vec.cwiseQuotient(normalization_weights);

	for (int i = 0; i < grid.get_nb_vertices(); ++i)
	{
		parameters[i] = splat_in_vec(i);
	}
}

cv::Mat fast_bilateral_solver::solve(cv::Mat initial_image_float, const bilateral_grid_simplified& grid, const std::vector<int>& data_cost_lookup, const int disparity_range, const float lambda, const int max_num_iterations, const int keep_nb_of_intermediate_images)
{
	assert(initial_image_float.type() == CV_32FC1);

	const int nb_pixels = initial_image_float.cols * initial_image_float.rows;

	// fill in initial parameters/image
	double* parameters = new double[grid.get_nb_vertices()];
	splat_initial_image(grid, initial_image_float, parameters);

	// create smoothing matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> Dn, Dm;
	bistochastize(grid, Dn, Dm);
	const Eigen::SparseMatrix<float, Eigen::RowMajor> mat_CBC = Dn * grid.blur(Dn);
	const Eigen::SparseMatrix<float, Eigen::RowMajor> mat_C_CBC = Dm - mat_CBC;

	// setup solver & solve
	ceres::GradientProblemSolver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = max_num_iterations;
	ceres::GradientProblemSolver::Summary summary;
	fast_bilateral_problem* pr = new fast_bilateral_problem(mat_C_CBC, grid, data_cost_lookup, disparity_range, lambda);
	ceres::GradientProblem problem(pr);
	ceres::Solve(options, problem, parameters, &summary);

	std::cout << summary.FullReport() << "\n";

	// copy result in to an eigen vector
	Eigen::VectorXf eig_parameters(grid.get_nb_vertices());
	for (int i = 0; i < grid.get_nb_vertices(); ++i)
	{
		eig_parameters(i) = static_cast<float>(parameters[i]);
	}

	// slice the result!
	Eigen::MatrixXf eig_result = grid.get_slice_matrix() * eig_parameters;

	// to and opencv image
	cv::Mat result_image(initial_image_float.rows, initial_image_float.cols, CV_32FC1);
	memcpy(result_image.data, eig_result.data(), sizeof(float) * nb_pixels);

	// process intermediate results for debugging
	{
		intermediate_results.clear();
		intermediate_results.resize(pr->intermediate_results.size());
		Eigen::MatrixXf eig_result2;
		for (size_t i = 0; i < pr->intermediate_results.size(); ++i)
		{
			eig_result2 = grid.get_slice_matrix() * pr->intermediate_results[i];
			auto& img = intermediate_results[i];
			img.create(initial_image_float.rows, initial_image_float.cols, CV_32FC1);
			memcpy(img.data, eig_result2.data(), sizeof(float) * nb_pixels);
		}

	}


	delete[] parameters;

	return result_image;
}