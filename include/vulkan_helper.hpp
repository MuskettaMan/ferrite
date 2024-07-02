#pragma once

#include <vulkan/vulkan.hpp>
#include <magic_enum.hpp>
#include <string>

namespace util
{
    static void VK_ASSERT(vk::Result result, std::string_view message)
    {
        if(result == vk::Result::eSuccess)
            return;

        static std::string completeMessage{};
        completeMessage = "[] ";
        auto resultStr = magic_enum::enum_name(result);

        completeMessage.insert(0, resultStr);
        completeMessage.insert(completeMessage.size() - 1, message);

        throw std::runtime_error(completeMessage.c_str());
    }

    static void VK_ASSERT(VkResult result, std::string_view message)
    {
        VK_ASSERT(vk::Result(result), message);
    }
}
