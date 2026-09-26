<% unless type_name == 'robject' %>
<% indexer_dims(true).each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)(p2) = m_<%=name%>(*(dtype*)(p1));
    }
}
<% end %>

// Moves 16 bytes per thread; see cumo_na_indexer_vec_row.
__global__ void <%="cumo_#{c_iter}_vec_kernel"%>(char* p1, char* p2, uint32_t n)
{
    typedef cumo_na_vec16_t<dtype> vec_t;
    const uint32_t vlen = sizeof(vec_t) / sizeof(dtype);
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * vlen;

    if (i + vlen <= n) {
        vec_t u = *(vec_t*)(p1 + (size_t)i * sizeof(dtype));
        vec_t z;
#pragma unroll
        for (uint32_t k = 0; k < vlen; ++k) z.v[k] = m_<%=name%>(u.v[k]);
        *(vec_t*)(p2 + (size_t)i * sizeof(dtype)) = z;
    } else {
        for (; i < n; ++i) ((dtype*)p2)[i] = m_<%=name%>(((dtype*)p1)[i]);
    }
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    const cumo_na_iarray_t* const arrays[] = {a1, a2};
    unsigned bcast;
    size_t grid_dim, block_dim;

    if (cumo_na_indexer_vec_row(indexer, arrays, 2, sizeof(dtype), &bcast) && !bcast) {
        uint64_t vlen = sizeof(cumo_na_vec16_t<dtype>) / sizeof(dtype);
        uint64_t threads = (indexer->total_size + vlen - 1) / vlen;
        <%="cumo_#{c_iter}_vec_kernel"%><<<(threads + CUMO_MAX_BLOCK_DIM - 1) / CUMO_MAX_BLOCK_DIM, CUMO_MAX_BLOCK_DIM, 0, cumo_cuda_stream()>>>(
            a1->ptr, a2->ptr, (uint32_t)indexer->total_size);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }
    grid_dim = cumo_get_grid_dim(indexer->total_size);
    block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_kernel", "*a1,*a2,*indexer", narrow: %w[a1 a2]) %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
