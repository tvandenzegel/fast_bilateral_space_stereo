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

#define ENABLE_DOMAIN_TRANSFORM_FILTER // you need OpenCV extra modules for this, you can just comment this otherwise.

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#ifdef ENABLE_DOMAIN_TRANSFORM_FILTER
#include <opencv2/ximgproc.hpp> 
#endif

#include "bilateral_grid_simplified.h"
#include "stereo_matcher_birchfield_tomasi.h"
#include "fast_bilateral_solver.h"

#define USE_EXAMPLE_1
//#define USE_EXAMPLE_2

int main(int argc, char** argv)
{
	/// EXAMPLE 1
	///
	///
#ifdef USE_EXAMPLE_1
	const std::string image_pair_filename = "../data/middlebury_scenes2006_midd1.jpg";

	// bilateral grid properties
	const int property_grid_sigma_spatial = 16;
	const int property_grid_sigma_luma = 16;
	const int property_grid_sigma_chroma = 16;

	// stereo matching properties
	const int property_disparity_min = -50;
	const int property_disparity_max = 50;
	const stereo_matcher_birchfield_tomasi::block_filter_size property_stereo_block_filter = stereo_matcher_birchfield_tomasi::block_filter_size::size_5x5;

	// solver properties
	const int property_solver_nb_iterations = 500;
	const float property_solver_lambda = 0.2f;
	const int property_solver_keep_nb_of_intermediate_images = 0; // you can ignore this one, for debugging
																  // post-process domain transform properties
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;
#endif
	///
	///
	///

	/// EXAMPLE 2
	///
	///
#ifdef USE_EXAMPLE_2
	const std::string image_pair_filename = "http://urixblog.com/p/2012/2012.09.13me/picture-7.jpg";

	// bilateral grid properties
	const int property_grid_sigma_spatial = 18;
	const int property_grid_sigma_luma = 16;
	const int property_grid_sigma_chroma = 24;

	// stereo matching properties
	const int property_disparity_min = -18;
	const int property_disparity_max = 18;
	const stereo_matcher_birchfield_tomasi::block_filter_size property_stereo_block_filter = stereo_matcher_birchfield_tomasi::block_filter_size::size_5x5;

	// solver properties
	const int property_solver_nb_iterations = 50;
	const float property_solver_lambda = 0.4f;
	const int property_solver_keep_nb_of_intermediate_images = 0; // you can ignore this one, for debugging

	// post-process domain transform properties
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;
#endif
	///
	///
	///

	// load stereo pair
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

	cv::imshow("stereo image pair", pair_image);
	cv::waitKey(16);

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
	// and let's use the minimum disparity image as a starting point.
	cv::Mat input_x = stereo_matcher.get_output().min_disp_image - stereo_matcher.get_parameters().disparity_min;
	cv::Mat input_x_fl;
	input_x.convertTo(input_x_fl, CV_32FC1);
	cv::Mat input_confidence_fl;
	stereo_matcher.get_output().conf_disp_image.convertTo(input_confidence_fl, CV_32FC1, 1.0f / 255.0f);

	// for initialization, let's apply a weighted bilateral filter!
	//   filtered image = blur(image x confidence) / blur(confidence)
	//   the confidence image is an image where a 1 means we have a match with the stereo matcher. 0 if there was no match.
	cv::Mat tc_im;
	cv::multiply(input_x_fl, input_confidence_fl, tc_im); 
	cv::Mat tc = grid.filter(tc_im);
	cv::Mat c = grid.filter(input_confidence_fl);
	cv::Mat start_point_image;
	cv::divide(tc, c, start_point_image);
	
	//// decomment if you want to start with 0...
	////start_point_image = cv::Scalar(0.f);

	cv::Mat final_disparty_image = solver.solve(start_point_image, grid, lookup,
		(stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min) + 1, property_solver_lambda, property_solver_nb_iterations, property_solver_keep_nb_of_intermediate_images);
	final_disparty_image += (float)stereo_matcher.get_parameters().disparity_min;

	// display disparity image
	cv::Mat adjmap_final;
	final_disparty_image.convertTo(adjmap_final, CV_8UC1,
		255.0 / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min), 
		-stereo_matcher.get_parameters().disparity_min * 255.0f / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min));
	cv::imshow("disparity image", adjmap_final);

	// optional: apply domain transform to smoothen the disparity image
#ifdef ENABLE_DOMAIN_TRANSFORM_FILTER
	cv::Mat final_disparty_dtfiltered_image;
	cv::ximgproc::dtFilter(stereo_images[0],
		final_disparty_image, final_disparty_dtfiltered_image,
		property_dt_sigmaSpatial, property_dt_sigmaColor,
		cv::ximgproc::DTF_RF,
		property_dt_numIters);

	// display disparity image
	cv::Mat adjmap_dt;
	final_disparty_dtfiltered_image.convertTo(adjmap_dt, CV_8UC1,
		255.0 / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min),
		-stereo_matcher.get_parameters().disparity_min * 255.0f / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min));
	cv::imshow("disparity image + domain transform", adjmap_dt);
#endif

	cv::waitKey(0);

	return 0;
}