#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWindowWidth = 960;
constexpr uint32_t kWindowHeight = 640;
constexpr int kFramesInFlight = 2;
constexpr const char* kSwapchainExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
constexpr float kCameraRadius = 5.15f;
constexpr float kCameraYaw = 0.66f;
constexpr float kCameraPitch = 0.48f;
constexpr float kMouseSensitivity = 0.005f;
constexpr float kMinimumCameraDistance = 2.0f;
constexpr float kMaximumCameraDistance = 15.0f;

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(result) + ")");
    }
}

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanPyramid {
public:
    void run() {
        initWindow();
        try {
            initVulkan();
            mainLoop();
            cleanupVulkan();
        } catch (...) {
            cleanupVulkan();
            glfwDestroyWindow(window_);
            glfwTerminate();
            throw;
        }
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> depthImages_;
    std::vector<VkDeviceMemory> depthImageMemories_;
    std::vector<VkImageView> depthImageViews_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapchainFramebuffers_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkFence> imagesInFlight_;
    std::array<VkSemaphore, kFramesInFlight> imageAvailableSemaphores_{};
    std::array<VkSemaphore, kFramesInFlight> renderFinishedSemaphores_{};
    std::array<VkFence, kFramesInFlight> inFlightFences_{};
    size_t currentFrame_ = 0;
    bool framebufferResized_ = false;
    bool orthographicView_ = false;
    bool mouseDragging_ = false;
    int dragButton_ = GLFW_MOUSE_BUTTON_LEFT;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    float cameraYaw_ = kCameraYaw;
    float cameraPitch_ = kCameraPitch;
    float cameraDistance_ = kCameraRadius;
    glm::vec3 cameraTarget_{0.0f};

    static void framebufferSizeCallback(GLFWwindow* window, int, int) {
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        app->framebufferResized_ = true;
    }

    static void keyCallback(GLFWwindow* window, int key, int, int action, int) {
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS && (key == GLFW_KEY_P || key == GLFW_KEY_O)) {
            app->orthographicView_ = key == GLFW_KEY_O;
            glfwSetWindowTitle(window, app->orthographicView_
                                           ? "Vulkan Pyramid - Orthographic"
                                           : "Vulkan Pyramid - Perspective");
        } else if (action == GLFW_PRESS && key == GLFW_KEY_R) {
            app->cameraYaw_ = kCameraYaw;
            app->cameraPitch_ = kCameraPitch;
            app->cameraDistance_ = kCameraRadius;
            app->cameraTarget_ = glm::vec3(0.0f);
        } else if ((action == GLFW_PRESS || action == GLFW_REPEAT) &&
                   (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)) {
            app->zoomCamera(0.9f);
        } else if ((action == GLFW_PRESS || action == GLFW_REPEAT) &&
                   (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)) {
            app->zoomCamera(1.0f / 0.9f);
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
        if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_MIDDLE &&
            button != GLFW_MOUSE_BUTTON_RIGHT) {
            return;
        }
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            app->mouseDragging_ = true;
            app->dragButton_ = button;
            glfwGetCursorPos(window, &app->lastCursorX_, &app->lastCursorY_);
            const bool controlHeld =
                glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            if (button == GLFW_MOUSE_BUTTON_LEFT && controlHeld) {
                app->pickOrbitPivot(app->lastCursorX_, app->lastCursorY_);
            }
        } else if (action == GLFW_RELEASE && app->dragButton_ == button) {
            app->mouseDragging_ = false;
        }
    }

    static void cursorPositionCallback(GLFWwindow* window, double x, double y) {
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (!app->mouseDragging_) {
            return;
        }

        const double deltaX = x - app->lastCursorX_;
        const double deltaY = y - app->lastCursorY_;
        app->lastCursorX_ = x;
        app->lastCursorY_ = y;
        if (app->dragButton_ == GLFW_MOUSE_BUTTON_LEFT) {
            const bool controlHeld =
                glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            if (controlHeld) {
                app->panCamera(static_cast<float>(deltaX), static_cast<float>(deltaY));
            } else {
                app->cameraYaw_ -= static_cast<float>(deltaX) * kMouseSensitivity;
                app->cameraPitch_ = std::clamp(
                    app->cameraPitch_ + static_cast<float>(deltaY) * kMouseSensitivity,
                    -1.45f, 1.45f);
            }
        } else {
            app->panCamera(static_cast<float>(deltaX), static_cast<float>(deltaY));
        }
    }

    void zoomCamera(float factor) {
        cameraDistance_ = std::clamp(cameraDistance_ * factor,
                                      kMinimumCameraDistance, kMaximumCameraDistance);
    }

    glm::vec3 cameraPosition() const {
        const float horizontalRadius = cameraDistance_ * std::cos(cameraPitch_);
        return cameraTarget_ + glm::vec3(
            horizontalRadius * std::sin(cameraYaw_),
            cameraDistance_ * std::sin(cameraPitch_),
            horizontalRadius * std::cos(cameraYaw_));
    }

    glm::mat4 viewProjection() const {
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        const glm::mat4 view = glm::lookAt(cameraPosition(), cameraTarget_,
                                           glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection;
        if (orthographicView_) {
            const float halfHeight = 1.8f * cameraDistance_ / kCameraRadius;
            const float halfWidth = halfHeight * aspect;
            projection = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                                    0.1f, 20.0f);
        } else {
            projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 20.0f);
        }
        projection[1][1] *= -1.0f;
        return projection * view;
    }

    void pickOrbitPivot(double cursorX, double cursorY) {
        int windowWidth = 0;
        int windowHeight = 0;
        glfwGetWindowSize(window_, &windowWidth, &windowHeight);
        if (windowWidth <= 0 || windowHeight <= 0 ||
            swapchainExtent_.width == 0 || swapchainExtent_.height == 0) {
            return;
        }

        const float framebufferX =
            static_cast<float>(cursorX * swapchainExtent_.width / windowWidth);
        const float framebufferY =
            static_cast<float>(cursorY * swapchainExtent_.height / windowHeight);
        const float ndcX = 2.0f * framebufferX / swapchainExtent_.width - 1.0f;
        const float ndcY = 2.0f * framebufferY / swapchainExtent_.height - 1.0f;
        const glm::mat4 inverseViewProjection = glm::inverse(viewProjection());
        glm::vec4 nearPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
        glm::vec4 farPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
        if (std::abs(nearPoint.w) < 1e-6f || std::abs(farPoint.w) < 1e-6f) {
            return;
        }
        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;
        const glm::vec3 rayOrigin =
            orthographicView_ ? glm::vec3(nearPoint) : cameraPosition();
        const glm::vec3 rayDirection = glm::normalize(glm::vec3(farPoint - nearPoint));
 
        static const std::array<glm::vec3, 18> triangles = {{
            {0.0f, 0.9f, 0.0f}, {-0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, -0.7f},
            {0.0f, 0.9f, 0.0f}, {0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, 0.7f},
            {0.0f, 0.9f, 0.0f}, {0.7f, -0.7f, 0.7f}, {-0.7f, -0.7f, 0.7f},
            {0.0f, 0.9f, 0.0f}, {-0.7f, -0.7f, 0.7f}, {-0.7f, -0.7f, -0.7f},
            {-0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, 0.7f}, {0.7f, -0.7f, -0.7f},
            {-0.7f, -0.7f, -0.7f}, {-0.7f, -0.7f, 0.7f}, {0.7f, -0.7f, 0.7f}
        }};

        float nearestDistance = std::numeric_limits<float>::infinity();
        glm::vec3 nearestPoint{};
        bool hit = false;
        for (size_t i = 0; i < triangles.size(); i += 3) {
            const glm::vec3 edge1 = triangles[i + 1] - triangles[i];
            const glm::vec3 edge2 = triangles[i + 2] - triangles[i];
            const glm::vec3 crossDirection = glm::cross(rayDirection, edge2);
            const float determinant = glm::dot(edge1, crossDirection);
            if (std::abs(determinant) < 1e-6f) {
                continue;
            }
            const float inverseDeterminant = 1.0f / determinant;
            const glm::vec3 fromVertex = rayOrigin - triangles[i];
            const float u = glm::dot(fromVertex, crossDirection) * inverseDeterminant;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            const glm::vec3 crossFromVertex = glm::cross(fromVertex, edge1);
            const float v = glm::dot(rayDirection, crossFromVertex) * inverseDeterminant;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            const float distance =
                glm::dot(edge2, crossFromVertex) * inverseDeterminant;
            if (distance > 0.0f && distance < nearestDistance) {
                nearestDistance = distance;
                nearestPoint = rayOrigin + rayDirection * distance;
                hit = true;
            }
        }
        if (hit) {
            const glm::vec3 offset = cameraPosition() - nearestPoint;
            const float distance = glm::length(offset);
            if (distance <= 1e-6f) {
                return;
            }
            cameraTarget_ = nearestPoint;
            cameraDistance_ = std::clamp(distance, kMinimumCameraDistance,
                                         kMaximumCameraDistance);
            cameraYaw_ = std::atan2(offset.x, offset.z);
            cameraPitch_ = std::clamp(
                std::asin(std::clamp(offset.y / distance, -1.0f, 1.0f)),
                -1.45f, 1.45f);
        }
    }

    void panCamera(float deltaX, float deltaY) {
        const float horizontalRadius = cameraDistance_ * std::cos(cameraPitch_);
        const glm::vec3 cameraOffset(
            horizontalRadius * std::sin(cameraYaw_),
            cameraDistance_ * std::sin(cameraPitch_),
            horizontalRadius * std::cos(cameraYaw_));
        const glm::vec3 forward = glm::normalize(-cameraOffset);
        const glm::vec3 right =
            glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 cameraUp = glm::cross(right, forward);
        const float unitsPerPixel =
            orthographicView_
                ? (3.6f * cameraDistance_ / kCameraRadius) /
                      static_cast<float>(swapchainExtent_.height)
                : (2.0f * cameraDistance_ * std::tan(glm::radians(22.5f))) /
                      static_cast<float>(swapchainExtent_.height);
        cameraTarget_ += right * (-deltaX * unitsPerPixel) +
                         cameraUp * (deltaY * unitsPerPixel);
    }

    void initWindow() {
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("GLFW initialization failed");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window_ = glfwCreateWindow(static_cast<int>(kWindowWidth),
                                   static_cast<int>(kWindowHeight),
                                   "Vulkan Pyramid - Perspective (O)", nullptr, nullptr);
        if (window_ == nullptr) {
            glfwTerminate();
            throw std::runtime_error("GLFW could not create a window");
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, cursorPositionCallback);
    }

    void initVulkan() {
        createInstance();
        checkVk(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
                "glfwCreateWindowSurface");
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createSwapchainResources();
        createSyncObjects();
    }

    void mainLoop() {
        while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
            glfwPollEvents();
            drawFrame();
        }
        checkVk(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
    }

    void cleanupVulkan() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            cleanupSwapchainResources();
            for (int i = 0; i < kFramesInFlight; ++i) {
                if (imageAvailableSemaphores_[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
                }
                if (renderFinishedSemaphores_[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
                }
                if (inFlightFences_[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(device_, inFlightFences_[i], nullptr);
                }
            }
            if (commandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            }
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    void createInstance() {
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Vulkan Pyramid";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        uint32_t extensionCount = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        if (extensions == nullptr || extensionCount == 0) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
        }

        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;

        checkVk(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
    }

    QueueFamilies findQueueFamilies(VkPhysicalDevice candidate) const {
        QueueFamilies families;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, queueFamilies.data());

        for (uint32_t i = 0; i < count; ++i) {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                families.graphics = i;
            }
            VkBool32 supportsPresent = VK_FALSE;
            checkVk(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_,
                                                         &supportsPresent),
                    "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (supportsPresent == VK_TRUE) {
                families.present = i;
            }
            if (families.complete()) {
                break;
            }
        }
        return families;
    }

    bool hasSwapchainExtension(VkPhysicalDevice candidate) const {
        uint32_t count = 0;
        checkVk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr),
                "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> extensions(count);
        checkVk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count,
                                                     extensions.data()),
                "vkEnumerateDeviceExtensionProperties");
        return std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::string(extension.extensionName) == kSwapchainExtension;
        });
    }

    SwapchainSupport querySwapchainSupport(VkPhysicalDevice candidate) const {
        SwapchainSupport support;
        checkVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface_,
                                                          &support.capabilities),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        uint32_t count = 0;
        checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &count, nullptr),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        support.formats.resize(count);
        if (count > 0) {
            checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &count,
                                                         support.formats.data()),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR");
        }

        count = 0;
        checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &count, nullptr),
                "vkGetPhysicalDeviceSurfacePresentModesKHR");
        support.presentModes.resize(count);
        if (count > 0) {
            checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &count,
                                                              support.presentModes.data()),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR");
        }
        return support;
    }

    bool isDeviceSuitable(VkPhysicalDevice candidate) const {
        if (!findQueueFamilies(candidate).complete() || !hasSwapchainExtension(candidate)) {
            return false;
        }
        const SwapchainSupport support = querySwapchainSupport(candidate);
        return !support.formats.empty() && !support.presentModes.empty();
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        checkVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                "vkEnumeratePhysicalDevices");
        if (count == 0) {
            throw std::runtime_error("No Vulkan-capable GPU was found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        checkVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                "vkEnumeratePhysicalDevices");

        int bestScore = -1;
        for (VkPhysicalDevice candidate : devices) {
            if (!isDeviceSuitable(candidate)) {
                continue;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            int score = 0;
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score = 3;
            } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score = 2;
            } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
                score = 1;
            }
            if (score > bestScore) {
                bestScore = score;
                physicalDevice_ = candidate;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No suitable Vulkan GPU found (a graphics queue and swapchain support are required)");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
        std::cout << "Using Vulkan device: " << properties.deviceName << '\n';
    }

    void createLogicalDevice() {
        const QueueFamilies families = findQueueFamilies(physicalDevice_);
        const std::set<uint32_t> uniqueFamilies = {*families.graphics, *families.present};
        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.push_back(queueInfo);
        }

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &features;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = &kSwapchainExtension;
        checkVk(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_),
                "vkCreateDevice");

        vkGetDeviceQueue(device_, *families.graphics, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, *families.present, 0, &presentQueue_);
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
        if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED) {
            return {VK_FORMAT_B8G8R8A8_SRGB, formats.front().colorSpace};
        }
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.front();
    }

    VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(
        const VkSurfaceCapabilitiesKHR& capabilities) const {
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> preferences = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        };
        for (VkCompositeAlphaFlagBitsKHR alpha : preferences) {
            if ((capabilities.supportedCompositeAlpha & alpha) != 0) {
                return alpha;
            }
        }
        throw std::runtime_error("Surface has no supported composite alpha mode");
    }

    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        VkExtent2D extent{
            static_cast<uint32_t>(std::max(width, 0)),
            static_cast<uint32_t>(std::max(height, 0))
        };
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
        return extent;
    }

    void createSwapchainResources() {
        createSwapchain();
        createImageViews();
        createDepthResources();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }

    void createSwapchain() {
        const SwapchainSupport support = querySwapchainSupport(physicalDevice_);
        const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
        const VkExtent2D extent = chooseExtent(support.capabilities);
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 &&
            imageCount > support.capabilities.maxImageCount) {
            imageCount = support.capabilities.maxImageCount;
        }

        const QueueFamilies families = findQueueFamilies(physicalDevice_);
        const std::array<uint32_t, 2> queueFamilyIndices = {
            *families.graphics, *families.present
        };

        VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        createInfo.surface = surface_;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (families.graphics != families.present) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount =
                static_cast<uint32_t>(queueFamilyIndices.size());
            createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = chooseCompositeAlpha(support.capabilities);
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_TRUE;

        checkVk(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_),
                "vkCreateSwapchainKHR");
        checkVk(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr),
                "vkGetSwapchainImagesKHR");
        swapchainImages_.resize(imageCount);
        checkVk(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount,
                                        swapchainImages_.data()),
                "vkGetSwapchainImagesKHR");
        swapchainImageFormat_ = surfaceFormat.format;
        swapchainExtent_ = extent;
    }

    void createImageViews() {
        swapchainImageViews_.reserve(swapchainImages_.size());
        for (VkImage image : swapchainImages_) {
            VkImageViewCreateInfo createInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            createInfo.image = image;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapchainImageFormat_;
            createInfo.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
            };
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            VkImageView imageView = VK_NULL_HANDLE;
            checkVk(vkCreateImageView(device_, &createInfo, nullptr, &imageView),
                    "vkCreateImageView");
            swapchainImageViews_.push_back(imageView);
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) != 0 &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("No compatible Vulkan memory type was found");
    }

    VkFormat findDepthFormat() const {
        constexpr std::array<VkFormat, 3> candidates = {
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT
        };
        for (VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);
            if ((properties.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth attachment format was found");
    }

    void createDepthResources() {
        depthFormat_ = findDepthFormat();
        depthImages_.resize(swapchainImages_.size(), VK_NULL_HANDLE);
        depthImageMemories_.resize(swapchainImages_.size(), VK_NULL_HANDLE);
        depthImageViews_.resize(swapchainImages_.size(), VK_NULL_HANDLE);

        for (size_t i = 0; i < depthImages_.size(); ++i) {
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = depthFormat_;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            checkVk(vkCreateImage(device_, &imageInfo, nullptr, &depthImages_[i]),
                    "vkCreateImage");

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device_, depthImages_[i], &requirements);
            VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocateInfo.allocationSize = requirements.size;
            allocateInfo.memoryTypeIndex =
                findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            checkVk(vkAllocateMemory(device_, &allocateInfo, nullptr,
                                     &depthImageMemories_[i]),
                    "vkAllocateMemory");
            checkVk(vkBindImageMemory(device_, depthImages_[i], depthImageMemories_[i], 0),
                    "vkBindImageMemory");

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = depthImages_[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = depthFormat_;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            checkVk(vkCreateImageView(device_, &viewInfo, nullptr, &depthImageViews_[i]),
                    "vkCreateImageView");
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat_;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        const std::array<VkAttachmentDescription, 2> attachments = {
            colorAttachment, depthAttachment
        };
        VkRenderPassCreateInfo createInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        createInfo.pAttachments = attachments.data();
        createInfo.subpassCount = 1;
        createInfo.pSubpasses = &subpass;
        createInfo.dependencyCount = 1;
        createInfo.pDependencies = &dependency;
        checkVk(vkCreateRenderPass(device_, &createInfo, nullptr, &renderPass_),
                "vkCreateRenderPass");
    }

    std::vector<char> readShader(const std::string& name) const {
        const std::string path = std::string(SHADER_DIR) + "/" + name;
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open shader: " + path);
        }
        const auto size = file.tellg();
        if (size <= 0 || size % 4 != 0) {
            throw std::runtime_error("Shader has an invalid SPIR-V size: " + path);
        }
        std::vector<char> code(static_cast<size_t>(size));
        file.seekg(0);
        file.read(code.data(), size);
        if (!file) {
            throw std::runtime_error("Could not read shader: " + path);
        }
        return code;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) const {
        VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule),
                "vkCreateShaderModule");
        return shaderModule;
    }

    void createGraphicsPipeline() {
        const auto vertexCode = readShader("triangle.vert.spv");
        const auto fragmentCode = readShader("triangle.frag.spv");
        const VkShaderModule vertexModule = createShaderModule(vertexCode);
        VkShaderModule fragmentModule = VK_NULL_HANDLE;
        try {
            fragmentModule = createShaderModule(fragmentCode);

            const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {{
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main", nullptr},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", nullptr}
            }};

            VkPipelineVertexInputStateCreateInfo vertexInput{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
            };
            VkPipelineInputAssemblyStateCreateInfo inputAssembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
            };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewportState{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
            };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
            };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
            };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depthStencil{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
            };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencil.maxDepthBounds = 1.0f;

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo colorBlending{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
            };
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            const std::array<VkDynamicState, 2> dynamicStates = {
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
            };
            VkPipelineDynamicStateCreateInfo dynamicState{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
            };
            dynamicState.dynamicStateCount =
                static_cast<uint32_t>(dynamicStates.size());
            dynamicState.pDynamicStates = dynamicStates.data();

            VkPipelineLayoutCreateInfo layoutInfo{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
            };
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            pushConstantRange.size = sizeof(glm::mat4);
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pushConstantRange;
            checkVk(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_),
                    "vkCreatePipelineLayout");

            VkGraphicsPipelineCreateInfo pipelineInfo{
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO
            };
            pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo.pStages = shaderStages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = pipelineLayout_;
            pipelineInfo.renderPass = renderPass_;
            pipelineInfo.subpass = 0;
            checkVk(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                              nullptr, &graphicsPipeline_),
                    "vkCreateGraphicsPipelines");

        } catch (...) {
            if (fragmentModule != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, fragmentModule, nullptr);
            }
            vkDestroyShaderModule(device_, vertexModule, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, vertexModule, nullptr);
    }

    void createFramebuffers() {
        swapchainFramebuffers_.reserve(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            const std::array<VkImageView, 2> attachments = {
                swapchainImageViews_[i], depthImageViews_[i]
            };
            VkFramebufferCreateInfo createInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            createInfo.renderPass = renderPass_;
            createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            createInfo.pAttachments = attachments.data();
            createInfo.width = swapchainExtent_.width;
            createInfo.height = swapchainExtent_.height;
            createInfo.layers = 1;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            checkVk(vkCreateFramebuffer(device_, &createInfo, nullptr, &framebuffer),
                    "vkCreateFramebuffer");
            swapchainFramebuffers_.push_back(framebuffer);
        }
    }

    void createCommandPool() {
        const QueueFamilies families = findQueueFamilies(physicalDevice_);
        VkCommandPoolCreateInfo createInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        createInfo.queueFamilyIndex = *families.graphics;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        checkVk(vkCreateCommandPool(device_, &createInfo, nullptr, &commandPool_),
                "vkCreateCommandPool");
    }

    void createCommandBuffers() {
        commandBuffers_.resize(swapchainFramebuffers_.size());
        imagesInFlight_.assign(swapchainFramebuffers_.size(), VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
        };
        allocateInfo.commandPool = commandPool_;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        checkVk(vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers_.data()),
                "vkAllocateCommandBuffers");
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = commandBuffers_[imageIndex];
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        VkClearValue clearColor{};
        clearColor.color = {{0.035f, 0.045f, 0.075f, 1.0f}};
        VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderPassInfo.renderPass = renderPass_;
        renderPassInfo.framebuffer = swapchainFramebuffers_[imageIndex];
        renderPassInfo.renderArea.extent = swapchainExtent_;
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0] = clearColor;
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphicsPipeline_);

        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = static_cast<float>(swapchainExtent_.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = swapchainExtent_;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        const glm::mat4 mvp = viewProjection();
        vkCmdPushConstants(commandBuffer, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(mvp), &mvp);
        vkCmdDraw(commandBuffer, 18, 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < kFramesInFlight; ++i) {
            checkVk(vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                                      &imageAvailableSemaphores_[i]),
                    "vkCreateSemaphore");
            checkVk(vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                                      &renderFinishedSemaphores_[i]),
                    "vkCreateSemaphore");
            checkVk(vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]),
                    "vkCreateFence");
        }
    }

    void drawFrame() {
        checkVk(vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                                UINT64_MAX),
                "vkWaitForFences");

        uint32_t imageIndex = 0;
        const VkResult acquireResult = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_],
            VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            framebufferResized_ = false;
            recreateSwapchain();
            return;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            checkVk(acquireResult, "vkAcquireNextImageKHR");
        }

        if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
            checkVk(vkWaitForFences(device_, 1, &imagesInFlight_[imageIndex], VK_TRUE,
                                    UINT64_MAX),
                    "vkWaitForFences (swapchain image)");
        }
        checkVk(vkResetCommandBuffer(commandBuffers_[imageIndex], 0),
                "vkResetCommandBuffer");
        recordCommandBuffer(imageIndex);
        checkVk(vkResetFences(device_, 1, &inFlightFences_[currentFrame_]),
                "vkResetFences");
        imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];
        const VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphores_[currentFrame_];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores_[currentFrame_];
        checkVk(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]),
                "vkQueueSubmit");

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores_[currentFrame_];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult presentResult = vkQueuePresentKHR(presentQueue_, &presentInfo);
        if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR &&
            presentResult != VK_ERROR_OUT_OF_DATE_KHR) {
            checkVk(presentResult, "vkQueuePresentKHR");
        }

        if (acquireResult == VK_SUBOPTIMAL_KHR ||
            presentResult == VK_SUBOPTIMAL_KHR ||
            presentResult == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized_) {
            framebufferResized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % kFramesInFlight;
    }

    void recreateSwapchain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }
        checkVk(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
        cleanupSwapchainResources();
        createSwapchainResources();
    }

    void cleanupSwapchainResources() {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }
        if (!commandBuffers_.empty() && commandPool_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, commandPool_,
                                 static_cast<uint32_t>(commandBuffers_.size()),
                                 commandBuffers_.data());
            commandBuffers_.clear();
        }
        imagesInFlight_.clear();
        for (VkFramebuffer framebuffer : swapchainFramebuffers_) {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        swapchainFramebuffers_.clear();
        for (VkImageView imageView : depthImageViews_) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, imageView, nullptr);
            }
        }
        depthImageViews_.clear();
        for (VkImage image : depthImages_) {
            if (image != VK_NULL_HANDLE) {
                vkDestroyImage(device_, image, nullptr);
            }
        }
        depthImages_.clear();
        for (VkDeviceMemory memory : depthImageMemories_) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, memory, nullptr);
            }
        }
        depthImageMemories_.clear();
        if (graphicsPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
            graphicsPipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (renderPass_ != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_, renderPass_, nullptr);
            renderPass_ = VK_NULL_HANDLE;
        }
        for (VkImageView imageView : swapchainImageViews_) {
            vkDestroyImageView(device_, imageView, nullptr);
        }
        swapchainImageViews_.clear();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchainImages_.clear();
    }
};

}  // namespace

int main() {
    try {
        VulkanPyramid app;
        app.run();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
