/* Copyright 2017 Toon Van den Zegel. All Rights Reserved.								*/
/*                                                                                      */
/* This file is part of bohme_shading_constraint_filter.								*/
/* 																						*/
/* bohme_shading_constraint_filter is free software :									*/
/* you can redistribute it and / or modify											*/
/* it under the terms of the GNU General Public License as published by					*/
/* the Free Software Foundation, either version 3 of the License, or					*/
/* (at your option) any later version.													*/
/* 																						*/
/* bohme_shading_constraint_filter is distributed in the hope that it will be useful,	*/
/* but WITHOUT ANY WARRANTY; without even the implied warranty of						*/
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the							*/
/* GNU General Public License for more details.											*/
/* 																						*/
/* You should have received a copy of the GNU General Public License					*/
/* along with bohme_shading_constraint_filter.                                          */
/* If not, see <http://www.gnu.org/licenses/>.					                     	*/

#pragma once

#include <Eigen/Sparse>
#include "eigen_sparse_serialize.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <chrono>
#include <unordered_map>

// A naïve implementation of a simplified bilateral grid.
// There are many ways you can optimize this class.
class bilateral_grid_simplified
{
public:
	std::int32_t get_nb_vertices() const { return nb_vertices; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& get_splat_matrix() const { return mat_splat; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& get_slice_matrix() const { return mat_slice; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& get_sblur_matrix() const { return mat_blur; }
	const Eigen::MatrixXf& get_normalizer_matrix() const { return mat_normalizer; }

	std::int32_t get_reference_nb_pixels() const { return nb_vertices; }
	std::int32_t get_reference_width() const { return nb_vertices; }
	std::int32_t get_reference_height() const { return nb_vertices; }
public:
	bilateral_grid_simplified();

	void init(const cv::Mat reference_bgr, const int sigma_spatial = 32, const int sigma_luma = 32, const int sigma_chroma = 32);

	Eigen::MatrixXf blur(Eigen::MatrixXf& in);

	Eigen::SparseMatrix<float, Eigen::RowMajor> blur(const Eigen::SparseMatrix<float, Eigen::RowMajor>& in);

	cv::Mat filter(cv::Mat input_image);

	cv::Mat splat_slice(cv::Mat input_image);

private:
	std::int32_t nb_vertices;
	std::int32_t nb_reference_pixels;
	std::int32_t reference_width;
	std::int32_t reference_height;
	Eigen::SparseMatrix<float, Eigen::RowMajor> mat_splat;
	Eigen::SparseMatrix<float, Eigen::RowMajor> mat_slice;
	Eigen::SparseMatrix<float, Eigen::RowMajor> mat_blur;
	Eigen::MatrixXf mat_normalizer;
};
