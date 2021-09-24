#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

// double fu(double f, double u) {
// 	if (f == 0.0 && u == INFINITY) {
// 		return 0.5;
// 	}

// 	return f * u;
// }

// static PyObject *rlss_assign(PyObject *self, PyObject *args) {
// 	PyObject *p1, *p2, *p3, *ret;
// 	int i, j, k, idx, max_nodes, n_nodes_e, n_alignment_nodes, n_routes;
// 	int *n_nodes, *idx_r, *idx_i, *idx_j, **alignments;
// 	double **travel_time_e, *frequencies, *arc_frequencies, *arc_costs, **u, *f;
// 	bool *to_consider;

// 	if (!PyArg_ParseTuple(
// 			args, "O!O!O!",
// 			&PyList_Type, &p1,
// 			&PyList_Type, &p2,
// 			&PyList_Type, &p3)) {
// 		PyErr_SetString(PyExc_TypeError, "parameters must be lists.");
// 		return NULL;
// 	}

// 	n_routes = (int) PyList_Size(p1);
// 	max_nodes = (int) PyList_Size(p3);
// 	n_nodes_e = max_nodes * 2;
// 	n_alignment_nodes = 0;

// 	alignments = (int **) malloc(n_routes * sizeof(int *));
// 	frequencies = (double *) malloc(n_routes * sizeof(double));
// 	travel_time_e = (double **) malloc(n_nodes_e * sizeof(double *));
// 	n_nodes = (int *) malloc(max_nodes * sizeof(int));
// 	u = (double **) malloc(n_nodes_e * sizeof(double *));

// 	for (i = 0; i < n_routes; i++) {
// 		PyObject *route = PyList_GetItem(p1, i);
// 		int route_length = (int) PyList_Size(route);
		
// 		n_nodes[i] = route_length;
// 		n_alignment_nodes += route_length;
// 		alignments[i] = (int *) malloc(route_length * sizeof(int));

// 		for (j = 0; j < route_length; j++)
// 			alignments[i][j] = PyLong_AsLong(PyList_GetItem(route, j));

// 		frequencies[i] = PyFloat_AsDouble(PyList_GetItem(p2, i));
// 	}

// 	for (i=0; i < n_nodes_e; i++) {
// 		travel_time_e[i] = (double *) malloc(n_nodes_e * sizeof(double));

// 		if (i < max_nodes) {
// 			PyObject *row = PyList_GetItem(p3, i);

// 			for (j=0; j < max_nodes; j++) {
// 				travel_time_e[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
// 			}
// 		}
// 	}

// 	const int stride = n_alignment_nodes - n_routes;
// 	idx_r = (int *) malloc(stride * 3 *sizeof(int));
// 	idx_i = (int *) malloc(stride * 3 *sizeof(int));
// 	idx_j = (int *) malloc(stride * 3 *sizeof(int));
// 	arc_frequencies = (double *) malloc(stride * 3 *sizeof(double));
// 	arc_costs = (double *) malloc(stride * 3 *sizeof(double));

// 	idx = 0;
// 	for (i=0; i < n_routes; i++) {
// 		for (j=0; j < n_nodes[i] - 1; j++) {
// 			idx_r[idx] = i;
// 			idx_r[idx + stride] = i;
// 			idx_r[idx + 2 * stride] = i;

// 			idx_i[idx] = alignments[i][j];
// 			idx_i[idx + stride] = alignments[i][j] + max_nodes;
// 			idx_i[idx + 2 * stride] = alignments[i][j + 1];

// 			idx_j[idx] = alignments[i][j + 1];
// 			idx_j[idx + stride] = alignments[i][j];
// 			idx_j[idx + 2 * stride] = alignments[i][j + 1] + max_nodes;

// 			arc_frequencies[idx] = frequencies[i];
// 			arc_frequencies[idx + stride] = INFINITY;
// 			arc_frequencies[idx + 2 * stride] = frequencies[i];

// 			arc_costs[idx] = travel_time_e[alignments[i][j]][alignments[i][j + 1]];
// 			arc_costs[idx] = travel_time_e[alignments[i][j] + max_nodes][alignments[i][j]];
// 			arc_costs[idx] = travel_time_e[alignments[i][j + 1]][alignments[i][j + 1] + max_nodes];

// 			idx += 1;
// 		}
// 	}

// 	for (i=0; i < n_nodes_e; i++) {
// 		u[i] = (double *) malloc(n_nodes_e * sizeof(double));

// 		for (j=0; j < n_nodes_e; j++) {
// 			u[i][j] = INFINITY;
// 		}
// 	}

// 	for (i=max_nodes; i < n_nodes_e; i++) {
// 		u[i][i] = 0;
// 		f = (double *) malloc(n_nodes_e * sizeof(double));
// 		for (j=0; j < n_nodes_e; j++) {
// 			f[j] = 0.0;
// 		}

// 		to_consider = (bool *) malloc(stride * 3 * sizeof(bool));
// 		for (j=0; j < stride * 3; j++) {
// 			to_consider[j] = true;
// 		}

// 		for (j=0; j < stride * 3; j++) {
// 			double min_cost = INFINITY;
// 			int min_cost_idx = 0;

// 			/* Get min u + arc_costs */
// 			for (k=0; k < stride * 3; k++) {
// 				double cost = u[idx_j[k]][i] + arc_costs[k];
// 				if (to_consider[k] && cost < min_cost) {
// 					min_cost = cost;
// 					min_cost_idx = k;
// 				}
// 			}

// 			to_consider[min_cost_idx] = false;
// 			if (u[idx_i[min_cost_idx]][i] >= min_cost) {
// 				if (arc_frequencies[min_cost_idx] == INFINITY) {
// 					u[idx_i[min_cost_idx]][i] = min_cost;
// 					f[idx_i[min_cost_idx]] = arc_frequencies[min_cost_idx];
// 				} else {
// 					u[idx_i[min_cost_idx]][i] = (fu(f[idx_i[min_cost_idx]], u[idx_i[min_cost_idx]][i]) + arc_frequencies[min_cost_idx] * min_cost) / (f[idx_i[min_cost_idx]] + arc_frequencies[min_cost_idx]);
// 					f[idx_i[min_cost_idx]] += arc_frequencies[min_cost_idx];
// 				}
// 			}
// 		}

// 		free(f);
// 		free(to_consider);
// 	}

// 	ret = PyList_New(max_nodes);

// 	for (i=0; i < max_nodes; i++) {
// 		PyObject *row = PyList_New(max_nodes);
// 		for (j=0; j < max_nodes; j++) {
// 			PyObject *value = Py_BuildValue("d", u[i][j + max_nodes]);
// 			PyList_SetItem(row, j, value);
// 		}

// 		PyList_SetItem(ret, i, row);
// 	}

// 	alignments = (int **) malloc(n_routes * sizeof(int *));
// 	frequencies = (double *) malloc(n_routes * sizeof(double));
// 	travel_time_e = (double **) malloc(n_nodes_e * sizeof(double *));
// 	n_nodes = (int *) malloc(max_nodes * sizeof(int));
// 	u = (double **) malloc(n_nodes_e * sizeof(double *));

// 	free(alignments);
// 	free(frequencies);
// 	free(travel_time_e);
// 	free(n_nodes);
// 	free(u);
// 	free(idx_r);
// 	free(idx_i);
// 	free(idx_j);
// 	free(arc_frequencies);
// 	free(arc_costs);
	
// 	return ret;
// }


static PyObject *rlss_total_trip_time(PyObject *self, PyObject *args) {
	PyObject *p1, *p2, *p3, *ret;
	bool complete_trip = false;
	int **alignments, *n_nodes, i, j, n_routes, max_nodes;
	double **travel_time, *allocations;

	if (!PyArg_ParseTuple(
			args, "O!O!O!|p",
			&PyList_Type, &p1, 
			&PyList_Type, &p2,
			&PyList_Type, &p3,
			&complete_trip)) {

	    PyErr_SetString(PyExc_TypeError, "parameters must be lists.");
	    return NULL;
	}

	n_routes = (int) PyList_Size(p1);
	max_nodes = (int) PyList_Size(p3);

	alignments = (int **) malloc(n_routes * sizeof(int *));
	allocations = (double *) malloc(n_routes * sizeof(double));
	travel_time = (double **) malloc(max_nodes * sizeof(double *));
	n_nodes = (int *) malloc(max_nodes * sizeof(int));
	ret = PyList_New(n_routes);

	/* Parse allocations and alignments */
	for (i = 0; i < n_routes; i++) {
		PyObject *route = PyList_GetItem(p1, i);
		PyObject *n_alloc = PyList_GetItem(p2, i);
		int route_length = (int) PyList_Size(route);

		n_nodes[i] = route_length;
		alignments[i] = (int *) malloc(route_length * sizeof(int));
		allocations[i] = PyLong_AsDouble(n_alloc);

		for (j = 0; j < route_length; j++)
			alignments[i][j] = PyLong_AsLong(PyList_GetItem(route, j));
	}

	/* Parse travel time matrix */
	for (i = 0; i < max_nodes; i++) {
		PyObject *row = PyList_GetItem(p3, i);
		travel_time[i] = (double *) malloc(max_nodes * sizeof(double));

		for (j = 0; j < max_nodes; j++)
			travel_time[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
	}

	for (i = 0; i < n_routes; i++) {
		double trip_time = 0.0;

		for (j=1; j < n_nodes[i]; j++)
			trip_time += travel_time[alignments[i][j - 1] - 1][alignments[i][j] - 1];

		// printf("Route %d trip time: %.2f", i, trip_time);

		if (complete_trip && alignments[i][0] != alignments[i][n_nodes[i] - 1]) {
			// printf("Complete trip not implemented\n");
		}

		double frequency = 1.0 / (trip_time / allocations[i] + 1e-20);
		// PyObject* value = Py_BuildValue("d", frequency);
		// PyList_SetItem(ret, i, value);

		// printf("Route %d: %.2f\n", i, frequency);
	}        

	free(alignments);
	free(allocations);
	free(travel_time);
	free(n_nodes);

	return ret;
}

static PyMethodDef RLSSMethods[] = {
	{
		"total_trip_time",
		rlss_total_trip_time,
		METH_VARARGS,
		"Calculates total trip time and frequency of a route"
	},
	// {
	// 	"assign",
	// 	rlss_assign,
	// 	METH_VARARGS,
	// 	"Transit assignment"
	// },
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef rlssmodule = {
	PyModuleDef_HEAD_INIT,
	"rlss", /* Name of module */
	NULL, /* Module documentation */
	-1,
	RLSSMethods
};

PyMODINIT_FUNC PyInit_rlss(void) {
	return PyModule_Create(&rlssmodule);
}

int main(int argc, char *argv[]) {
	wchar_t *program = Py_DecodeLocale(argv[0], NULL);
	if (program == NULL) {
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}

	if (PyImport_AppendInittab("rlss", PyInit_rlss) == -1) {
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}

	Py_SetProgramName(program);

	Py_Initialize();

	PyObject *pmodule = PyImport_ImportModule("rlss");
	if (!pmodule) {
		PyErr_Print();
		fprintf(stderr, "Error: could not import module 'rlss'\n");
	}

	PyMem_RawFree(program);
	return 0;
}