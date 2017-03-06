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

#include "bilateral_grid_simplified.h"

#include <Eigen/Sparse>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <chrono>

// Class for the stereo matching term
// Please paragraph '4. An Efficient Stereo Data Term' in the 
// 'Fast Bilateral Space Stereo for Synthetic Defocus" paper.
class stereo_matcher_birchfield_tomasi
{
public:
	enum class block_filter_size
	{
		size_5x5,
		size_15x15,
		size_25x25
	};

	struct parameters
	{
		parameters():
			disparity_min(0),
			disparity_max(16),
			noise_epsilon(4),
			filter_size(block_filter_size::size_25x25)
		{

		}

		int disparity_min;
		int disparity_max;
		int noise_epsilon;
		block_filter_size filter_size;
	};
	parameters& get_parameters() { return current_parameters; }

	struct output
	{
		cv::Mat min_disp_image;
		cv::Mat max_disp_image;
		cv::Mat conf_disp_image;

		// Debugging images
		cv::Mat block_match_image;
		cv::Mat block_match_image1_x;
		cv::Mat block_match_image2_x;
		cv::Mat block_match_image1_y;
		cv::Mat block_match_image_final;
	};
	const output& get_output() { return current_output; }

public:
	stereo_matcher_birchfield_tomasi();
	~stereo_matcher_birchfield_tomasi();

	const output& get_output() { return current_output; }
	parameters& get_parameters() { return current_parameters; }

	// This function generates a min and max disparity image
	// - [in] stereo_images: uint8 grayscale stereo images
	// output: see get_output() 
	void stereo_match(cv::Mat stereo_images[2]);

	// Generates a lookup table explained in the paper
	// - [in] grid: the bilateral grid
	// - [out] out_lookup: the output, a lookup table, meaning a cost value per disparity per vertex
	void generate_data_loss_table(const bilateral_grid_simplified& grid, std::vector<int>& out_lookup);
	
private:
	void block_filter_horz_21012(const cv::Mat& in, cv::Mat& out);
	void block_filter_horz_1050510(const cv::Mat& in, cv::Mat& out);
	
	void block_filter_vert_21012(const cv::Mat& in, cv::Mat& out);
	void block_filter_vert_1050510(const cv::Mat& in, cv::Mat& out);

	void block_filter_vert_505(const cv::Mat& in, cv::Mat& out);
	void block_filter_horz_505(const cv::Mat& in, cv::Mat& out);

	output current_output;
	parameters current_parameters;
};