import os
import pm4py
from pm4py.objects.petri_net.obj import PetriNet


def add_start_end_transitions(net, initial_marking, final_marking):
    
    N = len(net.transitions) + len(net.places) + 1
    t_start = PetriNet.Transition(name = 'n'+str(N), label='<START>')
    net.transitions.add(t_start)

    N = len(net.transitions) + len(net.places) + 1
    t_end = PetriNet.Transition(name = 'n'+str(N), label='<END>')
    net.transitions.add(t_end)

    N = len(net.transitions) + len(net.places) + 1
    new_place_start = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place_start)

    N = len(net.transitions) + len(net.places) + 1
    new_place_end = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place_end)

    places_from = list(initial_marking)
    places_to = list(final_marking)

    t_after_start = []
    for p in places_from:
        for arc in p.out_arcs:
            t_after_start.append(arc.target)
            net.arcs.remove(arc)
        p.out_arcs.clear()

    t_before_end = []
    for p in places_to:
        for arc in p.in_arcs:
            t_before_end.append(arc.source)
            net.arcs.remove(arc)
        p.in_arcs.clear()

    new_arcs = []

    for p in places_from:
        new_arcs.append(PetriNet.Arc(p, t_start))
        
    for p in places_to:
        new_arcs.append(PetriNet.Arc(t_end, p))

    new_arcs.append(PetriNet.Arc(t_start, new_place_start))
    new_arcs.append(PetriNet.Arc(new_place_end, t_end))

    for t in t_after_start:
        new_arcs.append(PetriNet.Arc(new_place_start, t))

    for t in t_before_end:
        new_arcs.append(PetriNet.Arc(t, new_place_end))

    for arc in new_arcs:
        net.arcs.add(arc)

    pm4py.write_pnml(net, initial_marking, final_marking, 'net_startEnd.pnml')
    net, initial_marking, final_marking = pm4py.read_pnml('net_startEnd.pnml')
    os.remove('net_startEnd.pnml')