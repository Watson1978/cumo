require_relative "../test_helper"

module Cumo::CUDA
  class CompilerTest < Test::Unit::TestCase
    sub_test_case "valid_kernel_name?" do
      def test_valid
        assert_true(Compiler.valid_kernel_name?('valid_name_1'))
      end

      def test_empty
        assert_false(Compiler.valid_kernel_name?(''))
      end

      def test_start_with_digit
        assert_false(Compiler.valid_kernel_name?('0_invalid'))
      end

      def test_new_line
        assert_false(Compiler.valid_kernel_name?("invalid\nname"))
      end

      def test_symbol
        assert_false(Compiler.valid_kernel_name?("invalid$name"))
      end

      def test_space
        assert_false(Compiler.valid_kernel_name?("invalid name"))
      end
    end

    sub_test_case "compile_using_nvrtc" do
      def test_valid
        compiler = Compiler.new
        source = "__global__ void k() {}\n"
        ptx = compiler.compile_using_nvrtc(source)
        assert { ptx =~ /Generated by NVIDIA NVVM Compiler/ }
      end

      def test_invalid
        compiler = Compiler.new
        source = "__global__ void k() {something_wrong}\n"
        assert_raise(CompileError) { compiler.compile_using_nvrtc(source) }
      end
    end

    sub_test_case "compile_with_cache" do
      CACHE_DIR = File.join(__dir__, '.kernel_cache')

      class << self
        def startup
          FileUtils.rm_rf(CACHE_DIR)
        end
      end

      def test_valid
        compiler = Compiler.new
        source = "__global__ void k() {}\n"
        assert_nothing_raised { compiler.compile_with_cache(source, cache_dir: CACHE_DIR) }
      end

      def test_valid_from_cache
        test_valid
        test_valid
      end
    end
  end
end
