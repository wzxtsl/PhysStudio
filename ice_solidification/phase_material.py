"""Spatially continuous ice/water optics from the simulated phase field."""
import bpy
import numpy as np


class PhaseMaterial:
    def __init__(self, data, scale):
        self.shape = tuple(int(x) for x in data['phase_shape'])
        self.origin = data['phase_origin'].copy()
        self.spacing = float(data['phase_spacing'])
        self.columns = int(data['phase_columns'])
        height, width = data['phase_atlas'].shape
        self.image = bpy.data.images.new('Simulated solid volume fraction', width, height,
                                         alpha=True, float_buffer=True)
        self.image.colorspace_settings.name = 'Non-Color'
        self.material = bpy.data.materials.new('Continuous freezing water')
        self.material.use_nodes = True
        self.tree = self.material.node_tree
        self.nodes, self.links = self.tree.nodes, self.tree.links
        p = self.nodes.get('Principled BSDF')
        p.inputs['Base Color'].default_value = (.97, .99, 1., 1.)
        p.inputs['Transmission Weight'].default_value = 1.
        p.inputs['Coat Weight'].default_value = .08
        position = self.nodes.new('ShaderNodeNewGeometry').outputs['Position']
        transform = self.nodes.new('ShaderNodeVectorMath'); transform.operation = 'SCALE'
        self.links.new(position, transform.inputs[0])
        transform.inputs['Scale'].default_value = 1. / (scale * self.spacing)
        offset = self.nodes.new('ShaderNodeVectorMath'); offset.operation = 'ADD'
        self.links.new(transform.outputs['Vector'], offset.inputs[0])
        offset.inputs[1].default_value = tuple(-self.origin / self.spacing)
        axes = self.nodes.new('ShaderNodeSeparateXYZ')
        self.links.new(offset.outputs[0], axes.inputs[0])
        x, y, z = [self.math('MINIMUM', self.math('MAXIMUM', axes.outputs[k], 0.), n - 1.)
                   for k, n in enumerate(self.shape)]
        z0 = self.math('FLOOR', z)
        z1 = self.math('MINIMUM', self.math('ADD', z0, 1.), self.shape[2] - 1.)
        values = []
        for layer in [z0, z1]:
            column = self.math('MODULO', layer, float(self.columns))
            row = self.math('FLOOR', self.math('DIVIDE', layer, float(self.columns)))
            u = self.math('DIVIDE', self.math('ADD', self.math('MULTIPLY', column, self.shape[0] + 2.),
                                             self.math('ADD', x, 1.5)), float(width))
            v = self.math('DIVIDE', self.math('ADD', self.math('MULTIPLY', row, self.shape[1] + 2.),
                                             self.math('ADD', y, 1.5)), float(height))
            uv = self.nodes.new('ShaderNodeCombineXYZ')
            self.links.new(u, uv.inputs['X']); self.links.new(v, uv.inputs['Y'])
            texture = self.nodes.new('ShaderNodeTexImage'); texture.image = self.image
            texture.interpolation = 'Linear'; texture.extension = 'EXTEND'
            self.links.new(uv.outputs[0], texture.inputs['Vector'])
            values.append(texture.outputs['Color'])
        weight = self.math('SUBTRACT', z, z0)
        phase = self.math('ADD', self.math('MULTIPLY', self.math('SUBTRACT', 1., weight), values[0]),
                          self.math('MULTIPLY', weight, values[1]))
        phase.node.label = 'Trilinear solid volume fraction'
        self.links.new(self.lerp(phase, 1.333, 1.309), p.inputs['IOR'])
        self.links.new(self.lerp(phase, .025, .14), p.inputs['Roughness'])
        absorb = self.nodes.new('ShaderNodeVolumeAbsorption')
        self.links.new(self.lerp(phase, .13, .22), absorb.inputs['Density'])
        color = self.nodes.new('ShaderNodeMixRGB'); color.blend_type = 'MIX'
        self.links.new(phase, color.inputs[0])
        color.inputs[1].default_value = (.85, .96, .99, 1.)
        color.inputs[2].default_value = (.72, .86, .95, 1.)
        self.links.new(color.outputs[0], absorb.inputs['Color'])
        noise = self.nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 5.
        self.links.new(position, noise.inputs['Vector'])
        ramp = self.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .35
        ramp.color_ramp.elements[0].color = (.20, .20, .20, 1.)
        ramp.color_ramp.elements[1].position = .75
        ramp.color_ramp.elements[1].color = (7., 7., 7., 1.)
        self.links.new(noise.outputs['Fac'], ramp.inputs[0])
        scatter = self.nodes.new('ShaderNodeVolumeScatter')
        scatter.inputs['Color'].default_value = (.85, .92, .95, 1.)
        scatter.inputs['Anisotropy'].default_value = .15
        density = self.math('MULTIPLY', self.math('POWER', phase, 2.), ramp.outputs['Color'])
        self.links.new(density, scatter.inputs['Density'])
        add = self.nodes.new('ShaderNodeAddShader')
        self.links.new(absorb.outputs[0], add.inputs[0]); self.links.new(scatter.outputs[0], add.inputs[1])
        self.links.new(add.outputs[0], self.nodes.get('Material Output').inputs['Volume'])
        detail = self.nodes.new('ShaderNodeTexNoise')
        detail.inputs['Scale'].default_value = 35.; detail.inputs['Detail'].default_value = 3.
        self.links.new(position, detail.inputs['Vector'])
        bump = self.nodes.new('ShaderNodeBump'); bump.inputs['Distance'].default_value = .006
        self.links.new(self.math('MULTIPLY', phase, .045), bump.inputs['Strength'])
        self.links.new(detail.outputs['Fac'], bump.inputs['Height'])
        self.links.new(bump.outputs['Normal'], p.inputs['Normal'])
        self.update(data)

    def math(self, operation, a, b=None):
        node = self.nodes.new('ShaderNodeMath'); node.operation = operation
        for i, value in enumerate([a, b]):
            if value is None:
                continue
            if isinstance(value, (float, int, np.number)):
                node.inputs[i].default_value = float(value)
            else:
                self.links.new(value, node.inputs[i])
        return node.outputs[0]

    def lerp(self, factor, a, b):
        return self.math('ADD', a, self.math('MULTIPLY', factor, b - a))

    def update(self, data):
        assert tuple(data['phase_shape']) == self.shape
        assert np.allclose(data['phase_origin'], self.origin)
        assert np.isclose(float(data['phase_spacing']), self.spacing)
        assert int(data['phase_columns']) == self.columns
        phase = data['phase_atlas']
        assert tuple(reversed(phase.shape)) == tuple(self.image.size)
        rgba = np.ones((*phase.shape, 4), np.float32)
        rgba[:, :, :3] = phase[:, :, None]
        self.image.pixels.foreach_set(rgba.ravel())
        self.image.update()
