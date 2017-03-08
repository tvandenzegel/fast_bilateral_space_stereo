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

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "bilateral_grid_simplified.h"
#include "stereo_matcher_birchfield_tomasi.h"
#include "fast_bilateral_solver.h"

int main(int argc, char** argv)
{
	const std::string image_pair_filename = "../data/middlebury_scenes2006_midd1.jpg";

	// bilateral grid properties
	const int property_grid_sigma_spatial = 18;
	const int property_grid_sigma_luma = 16;
	const int property_grid_sigma_chroma = 24;

	// stereo matching properties
	const int property_disparity_min = -30;
	const int property_disparity_max = 30;
	const stereo_matcher_birchfield_tomasi::block_filter_size property_stereo_block_filter = stereo_matcher_birchfield_tomasi::block_filter_size::size_5x5;

	// solver properties
	const int property_solver_nb_iterations = 1;
	const float property_solver_lambda = 0.5f;
	const int property_solver_keep_nb_of_intermediate_images = 0; // you can ignore this one, for debugging

	// post-process domain transform properties
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;



	cv::Mat stereo_images[2];
	cv::Mat pair_image;
	pair_image = cv::imread(image_pair_filename, CV_LOAD_IMAGE_COLOR);

	if (pair_image.cols > 0)
	{
		stereo_images[0] = pair_image(cv::Rect(0, 0, pair_image.cols >> 1, pair_image.rows)).clone();
		stereo_images[1] = pair_image(cv::Rect(pair_image.cols >> 1, 0, pair_image.cols >> 1, pair_image.rows)).clone();
	}
	else
	{
		std::cout << "failed to load " << image_pair_filename << std::endl;
		return -1;
	}

	// convert to gray scale
	cv::Mat stereo_images_gray[2];
	cv::cvtColor(stereo_images[0], stereo_images_gray[0], CV_BGR2GRAY);
	cv::cvtColor(stereo_images[1], stereo_images_gray[1], CV_BGR2GRAY);

	// grid
	bilateral_grid_simplified grid;
	grid.init(stereo_images[0], property_grid_sigma_spatial, property_grid_sigma_luma, property_grid_sigma_chroma);

	// stereo matching
	stereo_matcher_birchfield_tomasi stereo_matcher;
	stereo_matcher.get_parameters().disparity_min = property_disparity_min;
	stereo_matcher.get_parameters().disparity_max = property_disparity_max;
	stereo_matcher.get_parameters().filter_size = property_stereo_block_filter;
	stereo_matcher.stereo_match(stereo_images_gray);

	// loss function
	std::vector<int> lookup;
	stereo_matcher.generate_data_loss_table(grid, lookup);

	///// bilateral solver
	fast_bilateral_solver solver;

	// let's work from "0 --> disparity range" instead of "disparity min --> disparty max"
	cv::Mat input_x = stereo_matcher.get_output().min_disp_image - stereo_matcher.get_parameters().disparity_min;
	cv::Mat input_x_fl;
	input_x.convertTo(input_x_fl, CV_32FC1);
	cv::Mat input_confidence_fl;
	stereo_matcher.get_output().conf_disp_image.convertTo(input_confidence_fl, CV_32FC1, 1.0f / 255.0f);



	cv::Mat tc_im;
	cv::multiply(input_x_fl, input_confidence_fl, tc_im);
	cv::Mat tc = grid.filter(tc_im);
	cv::Mat c = grid.filter(input_confidence_fl);
	cv::Mat out;
	cv::divide(tc, c, out);
	out = out;
	////out = cv::Scalar(0.f);

	cv::Mat final_disparty_image = solver.solve(out, grid, lookup,
		(stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min) + 1, property_solver_lambda, property_solver_nb_iterations, property_solver_keep_nb_of_intermediate_images);
	final_disparty_image += (float)stereo_matcher.get_parameters().disparity_min;

	cv::Mat adjmap;
	final_disparty_image.convertTo(adjmap, CV_8UC1, 255.0 / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min), stereo_matcher.get_parameters().disparity_min);


	return 0;
}