# # Biblioteca do openCL
# import pyopencl as cl
# import numpy as np

# # Crear dos matrices de entrada y una matriz de salida
# a = np.random.rand(100, 100).astype(np.float32) + 1
# b = np.random.rand(100, 100).astype(np.float32) + 1 
# res = np.empty_like(a)
# plataformas = cl.get_platforms()
# ctx = cl.create_some_context()
# queue = cl.CommandQueue(ctx)

# mf = cl.mem_flags
# a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
# b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
# dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)

# # Modificar el kernel para realizar la multiplicación de matrices
# prg = cl.Program(ctx, """
#     __kernel void matrix_multiply(__global const float *a,
#     __global const float *b,
#     __global float *c,
#     const int N)
#     {
#       int gid_x = get_global_id(0);
#       int gid_y = get_global_id(1);
#       float sum = 0.0f;
#       for (int i = 0; i < N; i++) {
#         sum += a[gid_y * N + i] * b[i * N + gid_x];
#       }
#       c[gid_y * N + gid_x] = sum;
#     }
#     """).build()

# # Obtener el tamaño de las matrices
# N = a.shape[0]

# # Ejecutar el kernel
# prg.matrix_multiply(queue, a.shape, None, a_buf, b_buf, dest_buf, np.int32(N))

# # Copiar los resultados de vuelta a la matriz de salida
# cl.enqueue_copy(queue, res, dest_buf)
# print(plataformas)


# for platform in plataformas:
#     print("Plataforma:", platform.name)
    
#     devices = platform.get_devices()
    
#     for device in devices:
#         print("   Dispositivo:", device.name)
#         print("   Tipo do Dispositivo:", cl.device_type.to_string(device.type))
#         print("   Versão do OpenCL:", device.opencl_c_version)
#         print("   Memória Global Disponível (MB):", device.global_mem_size / (1024 * 1024))
#         print("   Número Máximo de Unidades de Computação:", device.max_compute_units)
#         print()

# print("Matriz A:")
# print(a)
# print("Matriz B:")
# print(b)
# print("Resultado de la multiplicación de matrices:")
# print(res)

import pyopencl as cl
import numpy as np
import cv2

# Função para processar uma imagem e modificar os valores de cores
def modify_image_colors(ctx, queue, image):
    mf = cl.mem_flags

    # Converte a imagem OpenCV para um formato adequado para OpenCL (RGBA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    h, w, c = image.shape
    image_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image)

    # Modifique o kernel OpenCL para aplicar as alterações de cor desejadas
    prg = cl.Program(ctx, """
        __kernel void modify_colors(__global uchar4* image, const int width, const int height)
        {
            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x < width && y < height) {
                int index = y * width + x;
                image[index].x += 10; // Aumenta o canal vermelho
                image[index].y += 10; // Aumenta o canal verde
                image[index].z += 10; // Aumenta o canal azul
                // O canal alpha (transparência) pode ser modificado se necessário
            }
        }
    """).build()

    # Execute o kernel para processar a imagem
    prg.modify_colors(queue, image.shape, None, image_buf, np.int32(w), np.int32(h))

    # Copie a imagem resultante de volta para a memória principal
    result = np.empty((h, w, c), dtype=np.uint8)
    cl.enqueue_copy(queue, result, image_buf)

    return result

# Carregue uma imagem usando OpenCV
image = cv2.imread('./88653a13a2fcca529087c9246c582b54.jpg')

# Configuração OpenCL
plataformas = cl.get_platforms()
for platform in plataformas:
    print("Plataforma:", platform.name)
    
    devices = platform.get_devices()
    
    for device in devices:
        print("   Dispositivo:", device.name)
        print("   Tipo do Dispositivo:", cl.device_type.to_string(device.type))
        print("   Versão do OpenCL:", device.opencl_c_version)
        print("   Memória Global Disponível (MB):", device.global_mem_size / (1024 * 1024))
        print("   Número Máximo de Unidades de Computação:", device.max_compute_units)
        print()
ctx = cl.create_some_context()
print(plataformas[0])
queue = cl.CommandQueue(ctx)

# Processar a imagem e modificar os valores de cores
result_image = modify_image_colors(ctx, queue, image)

# Exibir a imagem original e a imagem modificada (OpenCV)
cv2.imshow('Imagem Original', image)
cv2.imshow('Imagem Modificada', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar a imagem modificada (opcional)
cv2.imwrite('imagem_modificada.jpg', result_image)


