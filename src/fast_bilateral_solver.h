/* Copyright 2017 Toon Van den Zegel. All Rights Reserved.                              */
/*                                                                                      */
/* This file is part of fast_bilateral_space_stereo.                                    */
/*                                                                                      */
/* fast_bilateral_space_stereo is free software :                                       */
/* you can redistribute it and / or modify                                              */
/* it under the terms of the GNU General Public License as published by                 */
/* the Free Software Foundation, either version 3 of the License, or                    */
/* (at your option) any later version.                                                  */
/*                                                                                      */
/* fast_bilateral_space_stereo is distributed in the hope that it will be useful,       */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of                       */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the                          */
/* GNU General Public License for more details.                                         */
/*                                                                                      */
/* You should have received a copy of the GNU General Public License                    */
/* along with fast_bilateral_space_stereo.                                              */
/* If not, see <http://www.gnu.org/licenses/>.                                          */

#pragma once

#include <Eigen/Sparse>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <glog/logging.h>
#include <ceres/ceres.h>

#include <string>
#include <chrono>
#include <unordered_map>
#include <cassert>

#include "bilateral_grid_simplified.h"

// A naïve implementation of the bilateral solver.
// There are many ways you can optimize this class.
class fast_bilateral_solver
{
	class fast_bilateral_problem : public ceres::FirstOrderFunction
	{
	public:
		fast_bilateral_problem(
			const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat_C_CBC,
			const bilateral_grid_simplified& grid,
			const std::vector<int>& lookup,
			int disparty_range,
			float lambda,
			int keep_nb_of_intermediate_images);
		
		const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat_C_CBC;
		const bilateral_grid_simplified& grid;
		const std::vector<int>& lookup;
		const int disparty_range;
		const float lambda;
		const int nb_vertices;
		const float border_cost_value = 500.0f; // used if the disparity is beyond the lookup table range
		const int keep_nb_of_intermediate_images;

		mutable double intermediate_prev_energy;
		mutable std::vector<Eigen::VectorXf> intermediate_results;

		virtual int NumParameters() const override { return nb_vertices; }

		bool Evaluate(const double* const parameters,
			double* cost,
			double* gradient) const override;
	};


public:
	fast_bilateral_solver();
	
	// Bistochastize function
	// - See page 4 of "Fast Bilateral Space Stereo for Synthetic Defocus - Supplemental Material"
	void bistochastize(const bilateral_grid_simplified& in_grid, Eigen::SparseMatrix<float, Eigen::RowMajor>& out_Dn, Eigen::SparseMatrix<float, Eigen::RowMajor>& out_Dm, const int nb_iterations = 16);
	
	// Runs the solver
	// - [in] initial_image_float: initial values for the solver
	// - [in] bilateral_grid_simplified: 
	// - [in] data_cost_lookup: data cost lookup table, in this case the stereo disparity cost function
	// - [in] disparity_range: we work here from 0 ---> disparty_range
	// - [in] lambad: how much smoothing versus data term do you want?
	// - [in] max_num_iterations: number of iterations
	// - [in] keep_nb_of_intermediate_images: store the result of each iteration of the solver, one can set the maximum number of intermediate images it needs to store
	// returns the resolved image
	cv::Mat solve(cv::Mat initial_image_float, const bilateral_grid_simplified& grid, const std::vector<int>& data_cost_lookup, const int disparity_range, const float lambda, const int max_num_iterations = 25, const int keep_nb_of_intermediate_images = 0);

	// the result of each iteration (see keep_nb_of_intermediate_images in the solve function)
	const std::vector<cv::Mat>& get_intermediate_result_images() { return intermediate_results; }
private:
	// Splats and image in pixel space in to bilateral space
	void splat_initial_image(const bilateral_grid_simplified& grid, const cv::Mat initial_image_float, double* parameters);

	std::vector<cv::Mat> intermediate_results;
};