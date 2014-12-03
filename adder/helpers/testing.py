def test_truth_table(testmodel, input_ids, output_ids, truth_table,
         ht=0.8, lt=0.2, delay=99, plot=False):
    import tellurium as te
    r = te.loada(testmodel)

    sims = []

    for row in truth_table:
        message = ['When']
        for i, input_val in enumerate(row[0]):
            message.append(input_ids[i] + ':' + str(input_val))
            r[input_ids[i]] = input_val

        sim = r.simulate(0, delay + 1, delay + 1, ['time'] + input_ids + output_ids)
        sims.append(sim)

        for i, output_val in enumerate(row[1]):
            offset = len(input_ids) + 1 # Time + length of inputs
            ind = i + offset
            full_message = ' '.join(message + [
                'then',
                output_ids[i] + ':' + str(output_val),
                '; Found %s = %f' % (output_ids[i], sim[delay][ind])
            ])
            print full_message
            if output_val == 0:
                assert sim[delay][ind] < lt, full_message
            else:
                assert sim[delay][ind] > ht, full_message
    return r, sims
