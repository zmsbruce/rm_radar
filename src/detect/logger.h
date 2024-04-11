/**
 * @file logger.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file defines the Logger class to be used in NvInfer plugins.
 * @date 2024-04-11
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <NvInfer.h>

#include <iostream>

namespace radar::detect {

/**
 * @brief A class that implements the nvinfer1::ILogger interface for logging
 * messages with different severities.
 *
 */
class Logger : public nvinfer1::ILogger {
    using Severity = nvinfer1::ILogger::Severity;

   public:
    /**
     * @brief Constructs a Logger object with the specified severity level.
     *
     * @param severity The severity level for reporting messages. Defaults to
     * Severity::kWARNING.
     */
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportable_severity_(severity) {}

    /**
     * @brief Logs a message with the specified severity level.
     *
     * @param severity The severity level of the message.
     * @param msg The message to be logged.
     */
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportable_severity_) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[Fatal] " << msg << std::endl;
                std::abort();
            case Severity::kERROR:
                std::cerr << "[Error] " << msg << std::endl;
                std::abort();
            case Severity::kWARNING:
                std::cerr << "[Warning] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[Info] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[Verbose] " << msg << std::endl;
                break;
            default:
                std::cerr << "[Unknown] " << msg << std::endl;
        }
    }

   private:
    Severity reportable_severity_;
};

}  // namespace radar::detect