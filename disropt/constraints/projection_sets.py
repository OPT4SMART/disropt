import numpy as np


class AbstractSet:
    """Abstract constraint set
    """

    def projection(self, x):
        """Project a point onto the set
        """
        pass

    def to_constraints(self):
        """Convert the set in a list of Constraints
        """
        pass


class Box(AbstractSet):
    """Box set

    :math:`X = \\{x\\mid l\\leq x \\leq u\\}`
    with :math:`l,x,u\\in \\mathbb{R}^{n}`

    Args:
        lower_bound (numpy.ndarray, optional): array with lower bounds for each axis. Defaults to None.
        upper_bound (numpy.ndarray, optional): array with upper bounds for each axis. Defaults to None.

    Attributes:
        lower_bound (numpy.ndarray): lower bounds
        upper_bound (numpy.ndarray): upper bounds
        input_shape (tuple): space dimensions
    """

    def __init__(self, lower_bound=None, upper_bound=None):
        if lower_bound is None and upper_bound is None:
            raise ValueError("At least one argument must be provided.")

        if (lower_bound is not None):
            if not isinstance(lower_bound, np.ndarray):
                raise ValueError("lower_bound must be numpy.ndarray")
            elif len(lower_bound.shape) > 2 or lower_bound.shape[1] != 1:
                raise ValueError(
                    "inputs must be 2D numpy.ndarray with shape (n,1), n>=1")
            self.lower_bound = lower_bound

        if (upper_bound is not None):
            if not isinstance(upper_bound, np.ndarray):
                raise ValueError("upper_bound must be numpy.ndarray")
            elif len(upper_bound.shape) > 2 or upper_bound.shape[1] != 1:
                raise ValueError(
                    "inputs must be 2D numpy.ndarray with shape (n,1), n>=1")
            self.upper_bound = upper_bound

        if (lower_bound is not None) and (upper_bound is not None):
            if lower_bound.shape != upper_bound.shape:
                raise ValueError(
                    "Lower and upper bounds must have the same shape")
            else:
                self.input_shape = lower_bound.shape
        else:
            if lower_bound is None:
                self.lower_bound = -np.inf * np.ones(upper_bound.shape)
                self.input_shape = lower_bound.shape
            elif upper_bound is None:
                self.upper_bound = np.inf * np.ones(lower_bound.shape)
                self.input_shape = lower_bound.shape

    def intersection(self, box):
        """Compute the intersection with another box

        Args:
            box (Box): box to compute the intersection with

        Raises:
            ValueError: Only intersection with another box is supported
            ValueError: The two boxex must have the same input_shape
        """
        if not isinstance(box, Box):
            raise ValueError(
                "Only intersection with another box is supported")
        if self.input_shape != box.input_shape:
            raise ValueError(
                "Boxes must have the same shape {}.".format(self.input_shape))
        else:
            self.upper_bound = np.min([self.upper_bound, box.upper_bound], axis=0)
            self.lower_bound = np.max([self.lower_bound, box.lower_bound], axis=0)

    def projection(self, x):
        """Project a point onto the box

        Arguments:
            x (numpy.ndarray): Point to be projected

        Returns:
            numpy.ndarray: projected point
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("input must be a numpy.ndarray")
        if x.shape != self.input_shape:
            raise ValueError(
                "Dimensions mismatch. Required input shape is {}".format(self.input_shape))
        x_projected = np.zeros(self.input_shape)
        x_projected += x
        for i in range(self.input_shape[0]):
            if x[i] < self.lower_bound[i]:
                x_projected[i] = self.lower_bound[i]
            elif x[i] > self.upper_bound[i]:
                x_projected[i] = self.upper_bound[i]
        return x_projected

    def to_constraints(self):
        from ..functions import Variable
        x = Variable(self.input_shape[0])
        upper_bound = x <= self.upper_bound
        lower_bound = x >= self.lower_bound
        return [upper_bound, lower_bound]


class Strip(AbstractSet):
    """Strip set

    :math:`X = \\{x\\mid -w<=|a^\\top x - s|<=w\\}`
    with :math:`a,x\\in \\mathbb{R}^{n}` and :math:`w\\in\\mathbb{R}`

    Args:
        regressor (numpy.ndarray): regressor of the strip
        shift (float): shift from the origin
        width (float): width of the strip

    Attributes:
        regressor (numpy.ndarray): regressor of the strip
        shift (float): shift from the origin
        upper (float): upper border (shift + half width)
        lower (float): lower border (shift - half width)
        input_shape (tuple): space dimensions
    """

    def __init__(self, regressor, shift, width):
        if not isinstance(regressor, np.ndarray):
            raise ValueError("regressor must be numpy.ndarray")
        if len(regressor.shape) > 2 or regressor.shape[1] != 1:
            raise ValueError(
                "regressor must be 2D numpy.ndarray with shape (n,1), n>=1")
        self.regressor = regressor
        self.shift = shift
        self.upper = shift + width/2.0
        self.lower = shift - width/2.0
        self.input_shape = regressor.shape

    def __str__(self):
        description = r'{}<=|{}^T x - {}|<={}'.format(
            self.lower, self.regressor, self.shift, self.upper)
        return description

    def intersection(self, strip):
        """Compute the intersection with another strip

        Args:
            strip (Strip): strip to compute the intersection with

        Raises:
            ValueError: Only intersection with another strip is supported
            ValueError: The two strips must have the same regressor
        """
        if not isinstance(strip, Strip):
            raise ValueError(
                "Only intersection with another strip is supported")
        if self.regressor.any() != strip.regressor.any():
            raise ValueError(
                "Strips must have the same regressor {}.".format(self.regressor))
        else:
            self.upper = np.min([self.upper, strip.upper])
            self.lower = np.max([self.lower, strip.lower])

    def projection(self, x):
        """Project a point onto the strip

        Arguments:
            x (numpy.ndarray): Point to be projected

        Returns:
            numpy.ndarray: projected point
        """
        if self.input_shape != x.shape:
            raise ValueError("shape of the point {} is not equal to the shape of the regressor {}".format(
                x.shape, self.regressor.shape))
        distance = self.regressor.T.dot(x)
        if distance > self.upper:
            x_projected = x + float(self.upper - distance) * self.regressor
            return x_projected
        elif distance < self.lower:
            x_projected = x + float(self.lower - distance) * self.regressor
            return x_projected
        else:
            return x
    
    def to_constraints(self):
        from ..functions import Variable, Abs
        x = Variable(self.input_shape[0])
        upper_bound = Abs(self.regressor @ x - self.shift) <= self.upper
        lower_bound = Abs(self.regressor @ x - self.shift) >= self.lower
        return [upper_bound, lower_bound]


class Circle(AbstractSet):
    """Circle set

    :math:`X = \\{x\\mid \\|x - c\\|<= r\\}`
    with :math:`x,c\\in \\mathbb{R}^{n}` and :math:`r\\in\\mathbb{R}`

    Args:
        center (numpy.ndarray): center of the circle
        radius (float): radius of the circle

    Attributes:
        center (numpy.ndarray): center of the circle
        radius (float): radius of the circle
        input_shape (tuple): space dimensions
    """

    def __init__(self, center, radius):
        if not isinstance(center, np.ndarray):
            raise ValueError("center must be numpy.ndarray")
        if len(center.shape) > 2 or center.shape[1] != 1:
            raise ValueError(
                "center must be 2D numpy.ndarray with shape (n,1), n>=1")
        self.center = center
        self.radius = radius
        self.input_shape = center.shape

    def __str__(self):
        description = r'\|x - {}\|<={}'.format(
            self.center.flatten(), self.radius)
        return description

    def intersection(self, circle):
        """Compute the intersection with another circle

        Args:
            circle (Circle): circle to compute the intersection with

        Raises:
            ValueError: Only intersection with another circle is supported
            ValueError: The two circles must have the same center
        """
        if not isinstance(circle, Circle):
            raise ValueError(
                "Only intersection with another circle is supported")
        if self.center.any() != circle.center.any():
            raise ValueError(
                "Circles must have the same center {}.".format(self.center))
        else:
            self.radius = np.min([self.radius, circle.radius])

    def projection(self, x):
        """Project a point onto the circle

        Arguments:
            x (numpy.ndarray): Point to be projected

        Returns:
            numpy.ndarray: projected point
        """
        if self.input_shape != x.shape:
            raise ValueError("shape of the point {} does not comply with the dimension of the circle {}".format(
                x.shape, self.input_shape))
        distance = np.linalg.norm(self.center - x)
        if distance > self.radius:
            x_projected = self.center + self.radius * \
                (x - self.center) / distance
            return x_projected
        else:
            return x

    def to_constraints(self):
        from ..functions import Variable, Norm
        x = Variable(self.input_shape[0])
        f = Norm(x - self.center) <= self.radius
        return [f]


class CircularSector(AbstractSet):
    """Circular sector set.

    Args:
        vertex (numpy.ndarray): vertex of the circular sector
        angle (float): direction of the circular sector (in rad)
        radius (float): radius of the circular sector
        width (float): width of the circular sector (in rad)

    Attributes:
        vertex (numpy.ndarray): vertex of the circular sector
        angle (float): direction of the circular sector (in rad)
        radius (float): radius of the circular sector
        h_angle (float): left border (from the vertex pov) of the circular sector (in rad)
        l_angle (float): right border (from the vertex pov) of the circular sector (in rad)
        input_shape (tuple): space dimensions
    """

    def __init__(self, vertex, angle, radius, width):
        if not isinstance(vertex, np.ndarray):
            raise ValueError("vertex must be a numpy.ndarray")
        if len(vertex.shape) > 2 or vertex.shape[1] != 1:
            raise ValueError(
                "vertex must be 2D numpy.ndarray with shape (n,1), n>=1")
        if vertex.shape[0] != 2:
            raise ValueError("only dimension (2,1) is curtrently supported")
        self.vertex = vertex
        self.radius = radius
        self.angle = angle
        self.h_angle = self.angle + width/2.0
        self.l_angle = self.angle - width/2.0
        self.input_shape = vertex.shape

    def __str__(self):
        return "Circular sector"  # TODO

    def intersection(self, circular_sector):
        """Compute the intersection with another circular sector

        Args:
            circular_sector (CircularSector): circula sector to compute the intersection with

        Raises:
            ValueError: Only intersection with another circular_sector is supported
            ValueError: The two circular_sector must have the same vertex
        """
        if not isinstance(circular_sector, CircularSector):
            raise ValueError(
                "Only intersection with another circular_sector is supported")
        if self.vertex.any() != circular_sector.vertex.any():
            raise ValueError("Circular sectors must have the same vertex.")
        else:
            self.radius = np.min([self.radius, circular_sector.radius])
            sl = np.unwrap(self.l_angle)
            cl = np.unwrap(circular_sector.l_angle)

            if self.h_angle >= 0:
                self.h_angle = np.min([self.h_angle, circular_sector.h_angle])
            elif self.h_angle < 0:
                if circular_sector.h_angle < 0:
                    self.h_angle = np.min(
                        [self.h_angle, circular_sector.h_angle])
                else:
                    self.h_angle = circular_sector.h_angle

            if self.l_angle >= 0:
                if circular_sector.l_angle >= 0:
                    self.l_angle = np.max(
                        [self.l_angle, circular_sector.l_angle])
                else:
                    if cl-sl < np.pi:
                        self.l_angle = circular_sector.l_angle
            if self.l_angle < 0:
                if circular_sector.l_angle < 0:
                    self.l_angle = np.max(
                        [self.l_angle, circular_sector.l_angle])
                else:
                    if cl-sl > np.pi:
                        self.l_angle = circular_sector.l_angle
            self.h_angle = np.array([self.h_angle]).flatten()
            self.l_angle = np.array([self.l_angle]).flatten()

    def __set_bounds(self):
        """set the bounds of the circular sector
        """
        self.a = np.array([-np.sin(self.h_angle), np.cos(self.h_angle)])
        self.a = self.a/np.linalg.norm(self.a)
        self.b = np.array([-np.sin(self.l_angle), np.cos(self.l_angle)])
        self.b = self.b/np.linalg.norm(self.b)
        self.ca = np.dot(self.a.T, self.vertex)
        self.cb = np.dot(self.b.T, self.vertex)

    def projection(self, x):
        """Project a point onto the ciruclar sector

        Arguments:
            x (numpy.ndarray): Point to be projected

        Returns:
            numpy.ndarray: projected point
        """
        if self.input_shape != x.shape:
            raise ValueError(
                "shape of the point {} does not comply with the dimension of the circular sector {}".format(
                    x.shape, self.input_shape))
        self.__set_bounds()
        v = x-self.vertex
        # if in cone remain the same
        if np.dot(self.a.T, v) <= 0 and np.dot(self.b.T, v) >= 0:
            x_projected = x
        # if behind the projection is the vertex
        elif np.dot(self.a.T, v) > 0 and np.dot(self.b.T, v) < 0:
            x_projected = self.vertex
        elif np.dot(self.a.T, v) > 0 and np.dot(self.b.T, v) >= 0:
            aa = self.a
            cc = np.dot(aa.T, self.vertex)
            x_projected = x+((cc-np.dot(aa.T, x)) /
                             np.linalg.norm(aa, ord=2)**2)*aa
            if np.dot(self.b.T, x_projected-self.vertex) < 0:
                x_projected = self.vertex
        elif np.dot(self.a.T, v) <= 0 and np.dot(self.b.T, v) < 0:
            bb = self.b
            cc = np.dot(bb.T, self.vertex)
            x_projected = x+((cc-np.dot(bb.T, x)) /
                             np.linalg.norm(bb, ord=2)**2)*bb
            if np.dot(self.a.T, x_projected-self.vertex) > 0:
                x_projected = self.vertex
        else:  # TODO check
            x_projected = x

        distance = np.linalg.norm(x_projected-self.vertex, ord=2)
        if distance > self.radius:
            x_projected = self.vertex+self.radius * \
                (x_projected-self.vertex)/distance
        return x_projected

    def to_constraints(self):
        from ..functions import Variable, Norm
        x = Variable(self.input_shape[0])
        circular_bound = Norm(x - self.vertex) <= self.radius
        upper_bound = self.a @ (x - self.vertex) <= 0
        lower_bound = self.b @ (x - self.vertex) >= 0
        return [circular_bound, upper_bound, lower_bound]

