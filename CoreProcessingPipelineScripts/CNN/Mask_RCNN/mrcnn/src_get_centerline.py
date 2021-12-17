from itertools import combinations
import logging
import networkx as nx
from networkx.exception import NetworkXNoPath
import numpy as np
import operator
import scipy
from scipy.spatial import Voronoi
from scipy.ndimage import filters
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
#import matplotlib.pyplot as plt
from exceptions import CenterlineError

logger = logging.getLogger(__name__)


def get_centerline(
    geom,
    segmentize_maxlen=0.5,
    max_points=3000,
    simplification=0.05,
    smooth_sigma=5
):
    """
    Return centerline from geometry.

    Parameters:
    -----------
    geom : shapely Polygon or MultiPolygon
    segmentize_maxlen : Maximum segment length for polygon borders.
        (default: 0.5)
    max_points : Number of points per geometry allowed before simplifying.
        (default: 3000)
    simplification : Simplification threshold.
        (default: 0.05)
    smooth_sigma : Smoothness of the output centerlines.
        (default: 5)

    Returns:
    --------
    geometry : LineString or MultiLineString

    Raises:
    -------
    CenterlineError : if centerline cannot be extracted from Polygon
    TypeError : if input geometry is not Polygon or MultiPolygon

    """
    logger.debug("geometry type %s", geom.geom_type)

    if geom.geom_type == "Polygon":
        # segmentized Polygon outline
        #x0, y0 = geom.exterior.coords.xy
        #plt.plot(x0, y0, '.g', markersize=2)

        outline = _segmentize(geom.exterior, segmentize_maxlen)
        logger.debug("outline: %s", outline)

        # simplify segmentized geometry if necessary and get points
        outline_points = outline.coords

        #x1, y1 = outline_points.xy
        #plt.plot(x1, y1, marker='o', markersize=2)
        #plt.show()

        simplification_updated = simplification
        while len(outline_points) > max_points:
            #print('outline_points_while_A:', len(outline_points))
            # if geometry is too large, apply simplification until geometry
            # is simplified enough (indicated by the "max_points" value)
            simplification_updated += simplification
            outline_points = outline.simplify(simplification_updated).coords
            #print('outline_points_while_B:',  len(outline_points))
        logger.debug("simplification used: %s", simplification_updated)
        logger.debug("simplified points: %s", MultiPoint(outline_points))

        outline_points = _segment_post(outline_points, max_len = 15).coords
        _point_check(outline_points)
        # calculate Voronoi diagram and convert to graph but only use points
        # from within the original polygon
        vor = Voronoi(outline_points)
        #scipy.spatial.voronoi_plot_2d(vor)
        #plt.show()
        graph = _graph_from_voronoi(vor, geom)
        logger.debug(
            "voronoi diagram: %s", _multilinestring_from_voronoi(vor, geom)
        )

        # determine longest path between all end nodes from graph
        end_nodes = _get_end_nodes(graph)
        if len(end_nodes) < 2:
            logger.debug("Polygon has too few points")
            raise CenterlineError("Polygon has too few points")
        logger.debug("get longest path from %s end nodes", len(end_nodes))
        longest_paths = _get_longest_paths(end_nodes, graph)
        if not longest_paths:
            logger.debug("no paths found between end nodes")
            raise CenterlineError("no paths found between end nodes")
        if logger.getEffectiveLevel() <= 10:
            logger.debug("longest paths:")
            for path in longest_paths:
                logger.debug(LineString(vor.vertices[path]))

        # get least curved path from the five longest paths, smooth and
        # return as LineString
        centerline = _smooth_linestring(
            LineString(
                vor.vertices[_get_least_curved_path(
                    longest_paths, vor.vertices
                )]
            ), smooth_sigma
        )
        logger.debug("centerline: %s", centerline)
        logger.debug("return linestring")
        return centerline

    elif geom.geom_type == "MultiPolygon":
        logger.debug("MultiPolygon found with %s sub-geometries", len(geom))
        # get centerline for each part Polygon and combine into MultiLineString
        sub_centerlines = []
        for subgeom in geom:
            try:
                sub_centerline = get_centerline(
                    subgeom, segmentize_maxlen, max_points, simplification,
                    smooth_sigma
                )
                sub_centerlines.append(sub_centerline)
            except CenterlineError as e:
                logger.debug("subgeometry error: %s", e)
        # for MultPolygon, only raise CenterlineError if all subgeometries fail
        if sub_centerlines:
            return MultiLineString(sub_centerlines)
        else:
            raise CenterlineError("all subgeometries failed")

    else:
        raise TypeError(
            "Geometry type must be Polygon or MultiPolygon, not %s" %
            geom.geom_type
        )


# helper functions #
####################

def _segmentize(geom, max_len):
    """Interpolate points on segments if they exceed maximum length."""
    points = []
    for previous, current in zip(geom.coords, geom.coords[1:]):
        line_segment = LineString([previous, current])
        # add points on line segment if necessary
        points.extend([
            line_segment.interpolate(max_len * i).coords[0]
            for i in range(int(line_segment.length / max_len))
        ])
        # finally, add end point
        points.append(current)
    return LineString(points)

def _segment_post(outline_points, max_len):
    """Interpolate points on the longest segments after the simplification"""
    points = []
    distances = []
    for previous, current in zip(outline_points, outline_points[1:]):
        line_segment = LineString([previous, current])
        distances.append(line_segment.length)
        # add points on line segment if necessary

        points.extend([
            line_segment.interpolate(max_len * i).coords[0]
            for i in range(int(line_segment.length / max_len))
        ])

        # finally, add end point
        points.append(current)
    #print('max_distance:', np.max(distances))
    #print('distances',sorted(distances, reverse = True))
    return LineString(points)

def _point_check(outline_points):
    """Interpolate points on the longest segments"""

    distances = []
    for previous, current in zip(outline_points, outline_points[1:]):
        line_segment = LineString([previous, current])
        distances.append(line_segment.length)

    #print('max_distance:', np.max(distances))
    #print('final_points:', len(distances))
    #print('distances',sorted(distances, reverse = True))
    return distances


def _smooth_linestring(linestring, smooth_sigma):
    """Use a gauss filter to smooth out the LineString coordinates."""
    return LineString(
        zip(
            np.array(filters.gaussian_filter1d(linestring.xy[0], smooth_sigma)),
            np.array(filters.gaussian_filter1d(linestring.xy[1], smooth_sigma))
        )
    )


def _get_longest_paths(nodes, graph, maxnum=5):
    """Return longest paths of all possible paths between a list of nodes."""
    def _gen_paths_distances():
        for node1, node2 in combinations(nodes, r=2):
            try:
                yield nx.single_source_dijkstra(
                    G=graph, source=node1, target=node2, weight="weight"
                )
            except NetworkXNoPath:
                continue
    return [
        x for (y, x) in sorted(_gen_paths_distances(), reverse=True)
    ][:maxnum]


def _get_least_curved_path(paths, vertices):
    """Return path with smallest angles."""
    return min(
        zip([_get_path_angles_sum(path, vertices) for path in paths], paths),
        key=operator.itemgetter(0)
    )[1]


def _get_path_angles_sum(path, vertices):
    """Return all angles between edges from path."""
    return sum([
        _get_absolute_angle(
            (vertices[pre], vertices[cur]), (vertices[cur], vertices[nex])
        )
        for pre, cur, nex in zip(path[:-1], path[1:], path[2:])
    ])


def _get_absolute_angle(edge1, edge2):
    """Return absolute angle between edges."""
    v1 = edge1[0] - edge1[1]
    v2 = edge2[0] - edge2[1]
    return abs(
        np.degrees(np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
    )


def _get_end_nodes(graph):
    """Return list of nodes with just one neighbor node."""
    return [i for i in graph.nodes() if len(list(graph.neighbors(i))) == 1]


def _graph_from_voronoi(vor, geometry):
    """Return networkx.Graph from Voronoi diagram within geometry."""
    graph = nx.Graph()
    for x, y, dist in _yield_ridge_vertices(vor, geometry, dist=True):
        graph.add_nodes_from([x, y])
        graph.add_edge(x, y, weight=dist)
    return graph


def _multilinestring_from_voronoi(vor, geometry):
    """Return MultiLineString geometry from Voronoi diagram."""
    return MultiLineString([
        LineString([
            Point(vor.vertices[[x, y]][0]),
            Point(vor.vertices[[x, y]][1])
        ])
        for x, y in _yield_ridge_vertices(vor, geometry)
    ])


def _yield_ridge_vertices(vor, geometry, dist=False):
    """Yield Voronoi ridge vertices within geometry."""
    for x, y in vor.ridge_vertices:
        if x < 0 or y < 0:
            continue
        point1 = Point(vor.vertices[[x, y]][0])
        point2 = Point(vor.vertices[[x, y]][1])
        # Eliminate all points outside our geometry.
        if point1.within(geometry) and point2.within(geometry):
            if dist:
                yield x, y, point1.distance(point2)
            else:
                yield x, y
