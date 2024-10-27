/**
 * @file features.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief A file implementing a collection of feature vectors, which is used to
 * reduce times of memory collection and copy while appending like
 * `std::vector`.
 * @date 2024-04-09
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

namespace radar::track {

/**
 * @class Features
 * @brief A class to store and manage a collection of feature vectors.
 *
 * This class uses the Eigen library to handle feature vectors efficiently.
 * It allows dynamic resizing and provides methods to add new feature
 * vectors, access them, and manage the collection's capacity and size.
 */
class Features {
   public:
    Features() = default;

    /**
     * @brief Parameterized constructor for creating a Features object with a
     * specified size and capacity.
     * @param feature_size The size of each feature vector.
     * @param capacity Initial capacity of the feature collection (default is
     * 1).
     */
    Features(int feature_size, int capacity = 1)
        : features_(feature_size, capacity), capacity_(capacity), size_(0) {
        features_.setZero();
    }

    /**
     * @brief Constructor for creating a Features object from a single feature
     * vector.
     * @param feature The feature vector to initialize the collection with.
     * @param capacity Initial capacity of the feature collection (default is
     * 1).
     */
    Features(const Eigen::VectorXf& feature, int capacity = 1)
        : features_(feature.size(), capacity), capacity_(capacity), size_(1) {
        features_.setZero();
        features_.col(0) = feature;
    }

    Features(const Features& other)
        : features_(other.features_),
          capacity_(other.capacity_),
          size_(other.size_) {}

    Features(Features&& other) noexcept
        : features_(std::move(other.features_)),
          capacity_(other.capacity_),
          size_(other.size_) {
        other.capacity_ = 0;
        other.size_ = 0;
    }

    Features& operator=(const Features& other) {
        if (this != &other) {
            features_ = other.features_;
            capacity_ = other.capacity_;
            size_ = other.size_;
        }
        return *this;
    }

    Features& operator=(Features&& other) noexcept {
        if (this != &other) {
            features_ = std::move(other.features_);
            capacity_ = other.capacity_;
            size_ = other.size_;

            other.capacity_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Add a new feature vector to the collection.
     * @param feature The feature vector to add.
     * @throw std::runtime_error if the new feature's size does not match the
     * size of existing features.
     */
    void push_back(const Eigen::VectorXf& feature) {
        if (feature.rows() != features_.rows()) {
            throw std::runtime_error("row of feature is not the same");
        }

        if (size_ >= capacity_) {
            capacity_ *= 2;
            Eigen::MatrixXf new_features(features_.rows(), capacity_);
            new_features.setZero();
            new_features.block(0, 0, features_.rows(), features_.cols()) =
                features_;
            std::swap(features_, new_features);
        }
        features_.col(size_++) = feature;
    }

    /**
     * @brief Get a feature vector by index.
     * @param index The index of the feature vector to retrieve.
     * @return The feature vector at the specified index.
     * @throw std::out_of_range if the index is out of range.
     */
    inline Eigen::VectorXf get(int index) const {
        if (index < 0 || index >= size_) {
            throw std::out_of_range("index out of range");
        }
        return features_.col(index);
    }

    /**
     * @brief Get the entire collection of feature vectors as a matrix.
     * @return A constant reference to the internal matrix of feature vectors.
     */
    inline const Eigen::MatrixXf& get() const { return features_; }

    /**
     * @brief Get the number of feature vectors currently stored.
     * @return The size of the collection.
     */
    inline int size() const noexcept { return size_; }

    /**
     * @brief Get the current capacity of the collection.
     * @return The capacity of the collection.
     */
    inline int capacity() const noexcept { return capacity_; }

    /**
     * @brief Clears all the feature vectors from the collection.
     *
     * Resets the size of the collection to 0 and sets all elements in the
     * internal matrix to zero without changing the current capacity.
     */
    inline void clear() noexcept {
        size_ = 0;
        features_.setZero();
    }

    /**
     * @brief Get the rows of feature vectors currently stored.
     * @return The rows of the collection.
     */
    inline int rows() const noexcept { return features_.rows(); }

    /**
     * @brief Get the cols of feature vectors currently stored.
     * @return The cols of the collection.
     */
    inline int cols() const noexcept { return size_; }

    /**
     * @brief Gets the label of the feature vectors based on the maximum
     * coefficient in the feature sums.
     *
     * @return The label (index of the maximum coefficient) of the feature
     * vectors.
     */
    inline int label() const noexcept {
        Eigen::VectorXf sum = features_.rowwise().sum();
        int label;
        sum.maxCoeff(&label);
        return label;
    }

    /**
     * @brief Gets the normalized feature of the feature vectors.
     *
     * @return The normalized feature of the feature vectors.
     */
    inline Eigen::VectorXf feature() const noexcept {
        float sum = features_.sum();
        Eigen::VectorXf feature(features_.rows());
        if (iszero(sum)) {  // avoid division by zero
            feature.setZero();
        } else {
            feature = features_.rowwise().sum() / sum;
        }
        return feature;
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const Features& features) {
        os << features.features_.block(0, 0, features.rows(), features.cols());
        return os;
    }

   private:
    Eigen::MatrixXf features_;
    int capacity_ = 0;
    int size_ = 0;
};

}  // namespace radar::track