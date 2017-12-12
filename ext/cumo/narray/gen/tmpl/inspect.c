static VALUE
<%=c_iter%>(char *ptr, size_t pos, VALUE fmt)
{
<% if is_object %>
    return rb_inspect(*(VALUE*)(ptr+pos));
<% else %>
    return format_<%=type_name%>(fmt, (dtype*)(ptr+pos));
<% end %>
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
    return na_ndloop_inspect(ary, <%=c_iter%>, Qnil);
}
