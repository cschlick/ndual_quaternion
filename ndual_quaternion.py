"""
DualQuaternions operations, interpolation, conversions.
Author: Christopher Schlicksup
License: MIT
"""
import numpy as np
import quaternion  # numpy-quaternion
import json


class DualQuaternion:
    """
    This class is a vectorized modification of the dual quaternion class found here[1]
    It is heavily dependent on the numpy quaternion implementation from here[2]

    This is a very fast way to manipulate a large set of rigid transforms,
    much faster than with each individual transform as a Python instance and using for loops.

    Some initializations:
        1. DualQuaternion(rotational quaternion, dual quaternion)
        2. DualQuaternion.from_pose(rotational quaternion, translation vector)

    Examples:
        import numpy as np
        import quaternion
        from dual_quaternion import DualQuaternion

        # get some random rotations and translations
        rotation_quats = quaternion.from_float_array(np.random.random((100,4)))
        translations = np.random.random((100,3))

        # initialize the dual quaternion
        dq = DualQuaternion.from_pose(rotation_quats,translations,normalize=True)

        # transform the dual quaternion vector by another dual quaternion (here just using the first dual quaternion element)
        dq_tr = dq*dq[0]

        # get new translations
        dq_tr.translation # a ndarray of shape (100,3)

    Math operations:
        Multiplication (and a few other operations) are supported between dual quaternions of length: (1 with 1), (1 with N), (N with 1), or (N with N)
        But numpy-like broadcasting to higher dimensions will not work. (2 with 10) will result in an error.

        # to get delta dq
        delta_dq = start_dq.inverse * end_dq

        # compose sequential dqs
        result_dq = first_dq*second_dq


    [1] https://github.com/Achllle/dual_quaternions_ros
    [2] https://github.com/moble/quaternion
    """

    def __init__(self, q_r, q_d, normalize=False):

        if not (q_r.dtype == np.quaternion) or not (q_d.dtype == np.quaternion):
            raise ValueError("q_r and q_d must be of type np.quaternion. Instead received: {} and {}".format(
                type(q_r), type(q_d)))
        if (isinstance(q_r,np.ndarray) and isinstance(q_d,np.ndarray)) and (len(q_r)!=len(q_d)):
            raise ValueError(
                "q_r and q_d must be the same length. Instead received: {} and {}".format(len(q_r), len(q_d)))

        if normalize:
            self.q_d = q_d / np.norm(q_r)
            self.q_r = np.normalized(q_r)
        else:
            self.q_r = q_r
            self.q_d = q_d

        if isinstance(self.q_r, np.ndarray):
            self._len = len(self.q_r)
        else:
            self._len = 1
            self.shape = tuple([len(self)])

    def __str__(self):
        return "rotation: {}, translation: {}, \n".format(repr(self.q_r), repr(self.q_d)) + \
               "translation vector: {}".format(repr(self.translation()))

    def __repr__(self):
        return "<DualQuaternion: {0} + {1}e>".format(repr(self.q_r), repr(self.q_d))

    def __len__(self):
        return self._len

    def __getitem__(self,index):
        q_r_select = self.q_r[index]
        q_d_select = self.q_d[index]
        return self.__class__(q_r_select,q_d_select)

    def __mul__(self, other):
        """
        Dual quaternion multiplication
        :param other: right hand side of the multiplication: DualQuaternion instance
        :return product: DualQuaternion object. Math:
                      dq1 * dq2 = dq1_r * dq2_r + (dq1_r * dq2_d + dq1_d * dq2_r) * eps
        """
        q_r_prod = self.q_r * other.q_r
        q_d_prod = self.q_r * other.q_d + self.q_d * other.q_r
        product = DualQuaternion(q_r_prod, q_d_prod)

        return product

    def __imul__(self, other):
        """
        Dual quaternion multiplication with self-assignment: dq1 *= dq2
        See __mul__
        """
        return self.__mul__(other)

    def __rmul__(self, other):
        """Multiplication with a scalar
        :param other: scalar
        """
        return DualQuaternion(self.q_r * other, self.q_d * other)

    def __div__(self, other):
        """
        Dual quaternion division. See __truediv__
        :param other: DualQuaternion instance
        :return: DualQuaternion instance
        """
        return self.__truediv__(other)

    def __truediv__(self, other):
        """
        Dual quaternion division.
        :param other: DualQuaternion instance
        :return: DualQuaternion instance
        """
        other_r_sq = other.q_r * other.q_r
        prod_r = self.q_r * other.q_r / other_r_sq
        prod_d = (other.q_r * self.q_d - self.q_r * other.q_d) / other_r_sq

        return DualQuaternion(prod_r, prod_d)

    def __add__(self, other):
        """
        Dual Quaternion addition.
        :param other: dual quaternion
        :return: DualQuaternion(self.q_r + other.q_r, self.q_d + other.q_d)
        """
        return DualQuaternion(self.q_r + other.q_r, self.q_d + other.q_d)

    def __eq__(self, other):
        return (np.all(quaternion.isclose(self.q_r, other.q_r)) or np.all(quaternion.isclose(self.q_r,-other.q_r)))\
               and (np.all(quaternion.isclose(self.q_d, other.q_d)) or np.all(quaternion.isclose(self.q_d,-other.q_d)))

    def __ne__(self, other):
        return not self == other

    def transform_point(self, point_xyz):
        """
        Convenience function to apply the transformation to a given vector.
        dual quaternion way of applying a rotation and translation using matrices Rv + t or H[v; 1]
        This works out to be: sigma @ (1 + ev) @ sigma.combined_conjugate()
        If we write self = p + eq, this can be expanded to 1 + eps(rvr* + t)
        with r = p and t = 2qp* which should remind you of Rv + t and the quaternion
        transform_point() equivalent (rvr*)
        Does not check frames - make sure you do this yourself.
        :param point_xyz: list or np.array in order: [x y z]
        :return: vector of length 3
        """
        dq_point = DualQuaternion.from_dq_array([1, 0, 0, 0, 0, point_xyz[0], point_xyz[1], point_xyz[2]])
        res_dq = self * dq_point * self.combined_conjugate()

        return res_dq.dq_array()[5:]

    @classmethod
    def from_dq_array(cls, r_wxyz_t_wxyz):
        """
        Create a DualQuaternion instance from two quaternions in list format
        :param r_wxyz_t_wxyz: np.array or python list: np.array([q_rw, q_rx, q_ry, q_rz, q_tx, q_ty, q_tz]
        """
        r_wxyz_t_wxyz = np.asarray(r_wxyz_t_wxyz)
        if r_wxyz_t_wxyz.ndim>1:
            return cls(quaternion.as_float_array(r_wxyz_t_wxyz[:,:4]), quaternion.as_float_array(r_wxyz_t_wxyz[:,4:]))
        else:
            return cls(quaternion.as_float_array(r_wxyz_t_wxyz[:4]), quaternion.as_float_array(r_wxyz_t_wxyz[4:]))

    @classmethod
    def from_homogeneous_matrix(cls, arr):
        """
        Create a DualQuaternion instance from a 4 by 4 homogeneous transformation matrix
        :param arr: 4 by 4 list or np.array
        """
        q_r = quaternion.from_rotation_matrix(arr[:3, :3])
        quat_pose_array = np.zeros(7)
        quat_pose_array[:4] = np.array([q_r.w, q_r.x, q_r.y, q_r.z])

        quat_pose_array[4:] = arr[:3, 3]

        return cls.from_quat_pose_array(quat_pose_array)

    @classmethod
    def from_quat_pose_array(cls, r_wxyz_t_xyz):
        """
        Create a DualQuaternion object from an array of a quaternion r and translation t
        sigma = r + eps/2 * t * r
        :param r_wxyz_t_xyz: list in order: [q_rw, q_rx, q_ry, q_rz, tx, ty, tz]
        """

        q_r = np.quaternion(*r_wxyz_t_xyz[:4]).normalized()
        q_d = 0.5 * np.quaternion(0., *r_wxyz_t_xyz[4:]) * q_r
        return cls(q_r, q_d)


    @classmethod
    def from_pose(cls,*args,**kwargs):
        normalize = kwargs.get("normalize","True")
        if (len(args)==1) and (len(kwargs) in [0,1]) and isinstance(args[0],np.ndarray):
            return cls.from_quat_pose_array(args[0],normalize=normalize) # the original way
        elif (len(args)==2) and (len(kwargs) in [0,1]):
            a0, a1 = args[0], args[1]
            if not isinstance(a0,np.ndarray):
                a0 = np.array([a0])
            if not isinstance(a1,np.ndarray):
                a1 = np.array([a1])
            if a0.dtype==np.quaternion: # assume a0 is rotation quaternion, a1 is translation vector
                q_r = a0
                translation = a1
            elif a1.dtype==np.quaternion:
                q_r = a1
                translation = a0
            else:
                raise ValueError("At least one argument should be a rotation quaternion")
        elif (len(args)==0) and (len(kwargs)==2):
            if "rotq" in kwargs:
                q_r = kwargs.get("rotq")
            elif "q_r" in kwargs:
                q_r = kwargs.get("q_r")
            else:
                raise ValueError("Keyword arguments must inclue a rotation quaternion")
            if "translation" in kwargs:
                translation = kwargs.get("translation")
            elif "position" in kwargs:
                translation = kwargs.get("translation")
            else:
                raise ValueError("Keyword arguments must inclue a translation vector")
        else:
            raise ValueError("Invalid input: use DualQuaternion.from_pose(q_r, translation)")

        q_r = np.normalized(q_r)
        q_d_float = np.zeros((len(q_r),4))
        q_d_float[:,1:] = translation
        q_d = 0.5 * quaternion.from_float_array(q_d_float) * q_r
        return cls(q_r, q_d,normalize=normalize)



    @classmethod
    def from_translation_vector(cls, t_xyz):
        """
        Create a DualQuaternion object from a cartesian point
        :param t_xyz: list or np.array in order: [x y z]
        """
        return cls.from_quat_pose_array(np.append(np.array([1., 0., 0., 0.]), np.array(t_xyz)))

    @classmethod
    def identity(cls):
        return cls(quaternion.one, np.quaternion(0., 0., 0., 0.))



    def quaternion_conjugate(self):
        """
        Return the individual quaternion conjugates (qr, qd)* = (qr*, qd*)
        This is equivalent to inverse of a homogeneous matrix. It is used in applying
        a transformation to a line expressed in Plucker coordinates.
        See also DualQuaternion.dual_conjugate() and DualQuaternion.combined_conjugate().
        """
        return DualQuaternion(self.q_r.conjugate(), self.q_d.conjugate())

    def dual_number_conjugate(self):
        """
        Return the dual number conjugate (qr, qd)* = (qr, -qd)
        This form of conjugate is seldom used.
        See also DualQuaternion.quaternion_conjugate() and DualQuaternion.combined_conjugate().
        """
        return DualQuaternion(self.q_r, -self.q_d)

    def combined_conjugate(self):
        """
        Return the combination of the quaternion conjugate and dual number conjugate
        (qr, qd)* = (qr*, -qd*)
        This form is commonly used to transform a point
        See also DualQuaternion.dual_number_conjugate() and DualQuaternion.quaternion_conjugate().
        """
        return DualQuaternion(self.q_r.conjugate(), -self.q_d.conjugate())

    def inverse(self):
        """
        Return the dual quaternion inverse
        For unit dual quaternions dq.inverse() = dq.quaternion_conjugate()
        """
        q_r_inv = 1/self.q_r
        return DualQuaternion(q_r_inv, -q_r_inv * self.q_d * q_r_inv)

    def is_normalized(self):
        """Check if the dual quaternion is normalized"""
        if np.isclose(np.norm(self.q_r), 0):
            return True
        rot_normalized = np.isclose(np.norm(self.q_r), 1)
        trans_normalized = np.isclose(self.q_d / np.norm(self.q_r), self.q_d)
        return rot_normalized and trans_normalized

    def normalize(self):
        """
        Normalize this dual quaternion
        Modifies in place, so this will not preserve self
        """
        normalized = self.normalized()
        self.q_r = normalized.q_r
        self.q_d = normalized.q_d




    def pow(self, exponent):
        """self^exponent
        :param exponent: single float
        """
        exponent = float(exponent)

        q_r_float = quaternion.as_float_array(self.q_r)
        q_d_float = quaternion.as_float_array(self.q_d)

        if len(self)==1:
            q_r_w, q_r_vec = self.q_r.w, self.q_r.vec
            q_d_w, q_d_vec = self.q_d.w, self.q_d.vec

            q_r_w, q_r_vec = np.array([q_r_w])[:,None], np.array([q_r_vec])[:,None]
            q_d_w, q_d_vec = np.array([q_d_w])[:, None], np.array([q_d_vec])[:, None]


        else:
            q_r_float = quaternion.as_float_array(self.q_r)
            q_d_float = quaternion.as_float_array(self.q_d)
            q_r_w = q_r_float[:,0]
            q_r_vec = q_r_float[:,1:]
            q_d_w = q_d_float[:, 0]
            q_d_vec = q_d_float[:,1:]

        theta = 2 * np.arccos(q_r_w)



        if np.all(np.isclose(theta, 0)):
            return DualQuaternion.from_translation_vector(exponent * np.array(self.translation()))
        else:

            s0 = q_r_vec / np.sin(theta[:,None] / 2)
            d = -2. * q_d_w / np.sin(theta / 2)
            se = (q_d_vec - s0 * d[:,None] / 2 * np.cos(theta[:,None] / 2)) / np.sin(theta[:,None] / 2)

        q_r_float = np.zeros((len(self),4))
        q_r_float[:,0] = np.cos(exponent * theta / 2)
        q_r_float[:,1:] = np.sin(exponent * theta[:,None] / 2) * s0
        q_r = quaternion.from_float_array(q_r_float)

        q_d_float = np.zeros((len(self), 4))
        q_d_float[:,0] = -exponent * d / 2 * np.sin(exponent * theta / 2)
        #q_d_float[:,1:] = exponent * d / 2 * np.cos(exponent * theta / 2) * s0 + np.sin(exponent * theta / 2) * se
        p3 = (exponent * d /2 *np.cos(exponent * theta / 2))[:,None] * s0 + np.sin(exponent * theta[:,None] / 2) * se

        q_d_float[:,1:] = p3
        q_d = quaternion.from_float_array(q_d_float)
        return DualQuaternion(q_r, q_d)

    @classmethod
    def sclerp(cls, start, stop, t):
        """Screw Linear Interpolation
        Generalization of Quaternion slerp (Shoemake et al.) for rigid body motions
        ScLERP guarantees both shortest path (on the manifold) and constant speed
        interpolation and is independent of the choice of coordinate system.
        ScLERP(dq1, dq2, t) = dq1 * dq12^t where dq12 = dq1^-1 * dq2
        :param start: DualQuaternion instance
        :param stop: DualQuaternion instance
        :param t: fraction betweem [0, 1] representing how far along and around the
                  screw axis to interpolate
        """
        # ensure we always find closest solution. See Kavan and Zara 2005
        mult = start.q_r * stop.q_r
        if not isinstance(mult,np.ndarray):
            mult = np.array([mult])
        if len(start)>1:
            mult_w = quaternion.as_float_array(mult)[:,0]
            if np.any(mult_w<0):
                start.q_r[np.where(mult_w<0)]*=-1
        else:
            mult_w = mult[0].w
            if mult_w < 0:
                start.q_r *= -1

        return start * (start.inverse() * stop).pow(t)

    def nlerp(self, other, t):
        raise NotImplementedError()

    def save(self, path):
        """Save the transformation to file
        :param path: absolute folder path and filename + extension
        :raises IOError: when the path does not exist
        """
        with open(path, 'w') as outfile:
            json.dump(self.as_dict(), outfile)

    @classmethod
    def from_file(cls, path):
        """Load a DualQuaternion from file"""
        with open(path) as json_file:
            qdict = json.load(json_file)

        return cls.from_dq_array([qdict['r_w'], qdict['r_x'], qdict['r_y'], qdict['r_z'],
                                  qdict['d_w'], qdict['d_x'], qdict['d_y'], qdict['d_z']])

    def homogeneous_matrix(self):
        """Homogeneous 4x4 transformation matrix from the dual quaternion
        :return 4 by 4 np.array
        """
        homogeneous_mat = np.zeros([4, 4])
        rot_mat = quaternion.as_rotation_matrix(self.q_r)
        homogeneous_mat[:3, :3] = rot_mat
        homogeneous_mat[:3, 3] = np.array(self.translation())
        homogeneous_mat[3, 3] = 1.

        return homogeneous_mat

    def quat_pose_array(self):
        """
        Get the list version of the dual quaternion as a quaternion followed by the translation vector
        given a dual quaternion p + eq, the rotation in quaternion form is p and the translation in
        quaternion form is 2qp*
        :return: list [q_w, q_x, q_y, q_z, x, y, z]
        """
        return quaternion.as_float_array(self.q_r) + self.translation()

    def dq_array(self):
        """
        Get the float array version of the dual quaternion as the rotation quaternion followed by the translation quaternion
        :return: float array [q_rw, q_rx, q_ry, q_rz, q_tx, q_ty, q_tz]
        """
        if isinstance(self.q_r,np.quaternion):
            return np.array([self.q_r.w,self.q_r.x,self.q_r.y,self.q_r.z,self.q_d.w,self.q_r.x,self.q_r.y,self.q_r.z])

        elif isinstance(self.q_r,np.ndarray):
            if len(self)==1:
                dq_floats =  np.concatenate([quaternion.as_float_array(self.q_r[0]),quaternion.as_float_array(self.q_d[0])],axis=0)
            else:
                dq_floats = np.concatenate([quaternion.as_float_array(self.q_r), quaternion.as_float_array(self.q_d)],axis=1)

            return dq_floats


    def translation(self):
        """Get the translation component of the dual quaternion in vector form
        :return: float array [x y z]
        """
        mult = (2.0 * self.q_d) * self.q_r.conjugate()
        if isinstance(mult,np.ndarray) and len(mult)==1:
            mult = mult[0]
        t_q_float = quaternion.as_float_array(mult)
        if t_q_float.ndim==1:
            return t_q_float[1:]
        else:
            return t_q_float[:,1:]

    def normalized(self):
        """Return a copy of the normalized dual quaternion"""
        norm_qr = np.norm(self.q_r)
        return DualQuaternion(self.q_r / norm_qr, self.q_d / norm_qr)

    def as_dict(self):
        """dictionary containing the dual quaternion"""
        if len(self)>1:
            dq_array = self.dq_array()
            q_r_w, q_r_x, q_r_y, q_r_z, q_d_w, q_d_x, q_d_y, q_d_z = dq_array[:,0],dq_array[:,1],dq_array[:,2],dq_array[:,3],\
                                                                     dq_array[:,4],dq_array[:,5],dq_array[:,6],dq_array[:,7]
            return {'r_w': q_r_w, 'r_x': q_r_x, 'r_y': q_r_y, 'r_z': q_r_z,
                    'd_w': q_d_w, 'd_x': q_d_x, 'd_y': q_d_y, 'd_z': q_d_z}
        else:

            return {'r_w': self.q_r.w, 'r_x': self.q_r.x, 'r_y': self.q_r.y, 'r_z': self.q_r.z,
                    'd_w': self.q_d.w, 'd_x': self.q_d.x, 'd_y': self.q_d.y, 'd_z': self.q_d.z}

    def screw(self):
        """
        Get the screw parameters for this dual quaternion.
        Chasles' theorem (Mozzi, screw theorem) states that any rigid displacement is equivalent to a rotation about
        some line and a translation in the direction of the line. This line does not go through the origin!
        This function returns the Plucker coordinates for the screw axis (l, m) as well as the amount of rotation
        and translation, theta and d.
        If the dual quaternion represents a pure translation, theta will be zero and the screw moment m will be at
        infinity.
        :return: l (unit length), m, theta, d
        :rtype np.array(3), np.array(3), float, float
        """
        # start by extracting theta and l directly from the real part of the dual quaternion
        theta = self.q_r.angle()
        theta_close_to_zero = np.isclose(theta, 0)
        t = np.array(self.translation())

        if not theta_close_to_zero:
            l = self.q_r.vec / np.sin(theta / 2)  # since q_r is normalized, l should be normalized too

            # displacement d along the line is the projection of the translation onto the line l
            d = np.dot(t, l)

            # m is a bit more complicated. Derivation see K. Daniliidis, Hand-eye calibration using Dual Quaternions
            m = 0.5 * (np.cross(t, l) + np.cross(l, np.cross(t, l) / np.tan(theta / 2)))
        else:
            # l points along the translation axis
            d = np.linalg.norm(t)
            if not np.isclose(d, 0):  # unit transformation
                l = t / d
            else:
                l = (0, 0, 0)
            m = np.array([np.inf, np.inf, np.inf])

        return l, m, theta, d

    @classmethod
    def from_screw(cls, l, m, theta, d):
        """
        Create a DualQuaternion from screw parameters
        :param l: unit vector defining screw axis direction
        :param m: screw axis moment, perpendicular to l and through the origin
        :param theta: screw angle; rotation around the screw axis
        :param d: displacement along the screw axis
        """
        l = np.array(l)
        m = np.array(m)
        if not np.isclose(np.linalg.norm(l), 1):
            raise AttributeError("Expected l to be a unit vector, received {} with norm {} instead"
                                 .format(l, np.linalg.norm(l)))
        theta = float(theta)
        d = float(d)
        q_r = np.quaternion(np.cos(theta / 2), 0, 0, 0)
        q_r.vec = np.sin(theta / 2) * l
        q_d = np.quaternion(-d / 2 * np.sin(theta / 2), 0, 0, 0)
        q_d.vec = np.sin(theta / 2) * m + d / 2 * np.cos(theta / 2) * l

        return cls(q_r, q_d)