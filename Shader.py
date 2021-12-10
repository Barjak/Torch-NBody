from OpenGL.GL import *
import numpy as np

class Shader(object):
    def __init__(self):
        vertex_shader = shaders.compileShader("""
            #version 440
            uniform mat4 view;
            layout(std430, binding = 0) buffer obj_data1 {
                float obj_translate[];
            };
            layout(std430, binding = 1) buffer obj_data2 {
                float obj_scale[];
            };
            layout(std430, binding = 2) buffer obj_data3 {
                float obj_brightness[];
            };

            in vec3 position;
            out float distance;
            out float brightness;

            void main()
            {

                vec3 _t = vec3(obj_translate[(3*gl_InstanceID)],
                               obj_translate[(3*gl_InstanceID)+1],
                               obj_translate[(3*gl_InstanceID)+2]);
                mat4 obj_mat = mat4(1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    _t[0],_t[1],_t[2],1);
                mat4 scale_mat = mat4(obj_scale[gl_InstanceID],0,0,0,
                                      0,obj_scale[gl_InstanceID],0,0,
                                      0,0,obj_scale[gl_InstanceID],0,
                                      0,0,0,1);

                vec4 p = view * obj_mat * scale_mat * vec4(position, 1.0f);
                gl_Position = p;
                distance = p[2];
                brightness = obj_brightness[gl_InstanceID];
            }
            """, GL_VERTEX_SHADER)
        # print(glGetShaderiv(vertex_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(vertex_shader))

        fragment_shader = shaders.compileShader( """
            #version 440
            in float brightness;
            in float distance;
            out vec4 color;

            void main()
            {
                float l = 1. - log2(distance) / 12.0;
            //    color = vec4(l, l, l, 1.0f );
                color = vec4(brightness*l, brightness*l, brightness*l, 1.0f );
            }
            """ , GL_FRAGMENT_SHADER)
        # print(glGetShaderiv(fragment_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(fragment_shader))

        self._program = shaders.compileProgram(vertex_shader, fragment_shader);
        self._view = glGetUniformLocation(self._program, "view")
        self._position = glGetAttribLocation(self._program, "position")

    def use(self):
        glUseProgram(self._program)

    @property
    def view(self):
        return self._view
    @view.setter
    def view(self, matrix):
        glUniformMatrix4fv(self._view, 1, GL_TRUE, matrix)
    @property
    def position(self):
        return self._position
    @property
    def obj_translate(self):
        return 0
    @property
    def obj_scale(self):
        return 1
    @property
    def obj_brightness(self):
        return 2
