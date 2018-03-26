static VALUE
<%=c_iter%>(char *ptr, size_t pos, VALUE fmt)
{
    dtype x;
    LOAD_BIT(ptr,pos,x);
    return format_<%=type_name%>(fmt, x);
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
VALUE
<%=c_func(0)%>(VALUE ary)
{
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    return cumo_na_ndloop_inspect(ary, <%=c_iter%>, Qnil);
}
