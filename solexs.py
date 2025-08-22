import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SoLEXS_df = pd.read_parquet('/kaggle/input/isro-aditya-l1-solexs-lightcurve-fits/SoLEXS_dataset.parquet', engine='pyarrow')

display(SoLEXS_df)
display(SoLEXS_df.loc[:, ['DATE', 'TIME']].apply(['min', 'max'], axis=0))
display(SoLEXS_df.describe().loc[:, ['COUNTS']].T.astype(int))


start_date = '2025-07-27'
start_time = '00:00:00'

end_date   = '2025-08-22'
end_time   = '23:59:59'


gap = 60*30    # Seconds of inactivity to split events
sigma = 10     # Threshold = median + sigma * std



# Try to use DatePicker widgets
try:
    # Create proper date picker widgets
    date_widget = widgets.DatePicker(
        description='Start date:',
        value=pd.to_datetime(start_date).date()
    )

    end_date_widget = widgets.DatePicker(
        description='End date:',
        value=pd.to_datetime(end_date).date()
    )
except (ImportError, AttributeError, TypeError):
    # Fall back to text widgets if DatePicker isn't available
    date_widget = widgets.Text(
        description='Start date:',
        value=start_date,
        placeholder='YYYY-MM-DD'
    )

    end_date_widget = widgets.Text(
        description='End date:',
        value=end_date,
        placeholder='YYYY-MM-DD'
    )

# Time inputs (as text)
start_time_widget = widgets.Text(
    description='Start time:',
    value=start_time,
    placeholder='HH:MM:SS'
)

end_time_widget = widgets.Text(
    description='End time:',
    value=end_time,
    placeholder='HH:MM:SS'
)

# Add descriptions for parameter widgets
sigma_widget = widgets.IntSlider(
    value=sigma,
    min=1,
    max=10,
    step=1,
    description='Sigma:',
    tooltip='Threshold = median + sigma * std'
)

gap_widget = widgets.IntSlider(
    value=gap,
    min=60,
    max=7200,
    step=60,
    description='Gap (sec):',
    tooltip='Seconds of inactivity to split events'
)

# Add zoom window widget
zoom_window = 20  # Default: 10 minutes
zoom_widget = widgets.IntSlider(
    value=zoom_window,
    min=1,
    max=60,
    step=1,
    description='Zoom (min):',
    tooltip='Minutes to show around flare peak'
)

# Add a Run button
run_button = widgets.Button(
    description='Update Analysis',
    button_style='success',
    icon='play', 
    tooltip='Update parameters and data window',
    layout=widgets.Layout(margin='20px')
)

# Stats output area
stats_output = widgets.Output()

# Layout with styling
controls = widgets.VBox([
    widgets.HTML("<h3>Solar Flare Analysis Parameters</h3>"),
    widgets.HBox([
        widgets.VBox([date_widget, start_time_widget]),
        widgets.VBox([end_date_widget, end_time_widget])
    ]),
    widgets.HBox([
        widgets.VBox([sigma_widget]),
        widgets.VBox([gap_widget])
    ]),
    widgets.VBox([zoom_widget]),
    run_button,
    stats_output  # Add stats output directly below controls
])

# Function to run analysis
def run_analysis(b):
    global df, times, counts, med, std, flares, zoom_window
    
    # Update global zoom_window from widget
    zoom_window = zoom_widget.value
    
    # Get values from widgets
    if hasattr(date_widget.value, 'strftime'):
        start_date_val = date_widget.value.strftime('%Y-%m-%d')
    else:
        start_date_val = date_widget.value
        
    if hasattr(end_date_widget.value, 'strftime'):
        end_date_val = end_date_widget.value.strftime('%Y-%m-%d')
    else:
        end_date_val = end_date_widget.value
    
    start_time_val = start_time_widget.value
    end_time_val = end_time_widget.value
    sigma_val = sigma_widget.value
    gap_val = gap_widget.value
    
    # Clear previous output
    with stats_output:
        clear_output()
        
        # Show current parameters
        print(f"Analysis parameters:\n")
        print(f"Observation START: {start_date_val} {start_time_val}")
        print(f"Observation  END : {end_date_val} {end_time_val}")
        print(f"Sigma: {sigma_val} (flare detection threshold)")
        print(f"Gap: {gap_val} seconds (between separate flares)")
        print(f"Zoom window: {zoom_window} minutes (for flare detail view)")
        
        # Filter by date range
        df = SoLEXS_df[
            (SoLEXS_df['DATE'] >= start_date_val) & 
            (SoLEXS_df['DATE'] <= end_date_val)
        ].copy()
        
        # Convert time strings to seconds
        h, m, s = map(int, start_time_val.split(':'))
        start_index = h * 3600 + m * 60 + s
        
        h, m, s = map(int, end_time_val.split(':'))
        end_index = len(df) - (86400 - h * 3600 + m * 60 + s)
        
        # Apply time filtering
        df = df.iloc[start_index:end_index]
        
        if len(df) == 0:
            print("\nNo data available for the selected range.")
            return
        
        # Display data window
        print("\n\nData Window:")
        display(df.loc[:, ['DATE', 'TIME']].apply(['min', 'max'], axis=0))
        display(df.describe().loc[:, ['COUNTS']].T.astype(int))
        
        times = df['TIME']
        counts = df['COUNTS']
        
        # Find flares
        med, std = np.nanmedian(counts), np.nanstd(counts)
        mask = counts > (med + sigma_val * std)
        spikes = np.unique(times[mask])
        
        flares = []
        if spikes.size == 0:
            print("\nNo flares detected.")
        else:
            groups = [[spikes[0]]]
            for t in spikes[1:]:
                if t - groups[-1][-1] <= gap_val:
                    groups[-1].append(t)
                else:
                    groups.append([t])
            events = [(g[0], g[-1]) for g in groups]
            print(f"\nDetected {len(events)} {'flares' if len(events) > 1 else 'flare'}:\n")
            for i, ev in enumerate(events, 1):
                t0, t1 = Time(ev[0], format='unix'), Time(ev[1], format='unix')
                # midnight of that day:
                mid = Time(t0.iso[:10]+'T00:00:00', format='isot', scale='utc')
                info = {'start_iso': t0.iso[:-4],
                        'end_iso':   t1.iso[:-4],
                        'start_sod': (t0 - mid).sec,
                        'end_sod':   (t1 - mid).sec}
                flares.append(info)
                print(f"🔥 Flare {i}: {info['start_iso']} → {info['end_iso']}")



 # Light Curve Plot with Update Button
plot_output = widgets.Output()
update_plot_button = widgets.Button(
    description='Update',
    button_style='info',
    icon='refresh'
)

def update_light_curve(b):
    global df, times, counts, med, std, flares
    
    with plot_output:
        clear_output()
        
        if 'df' not in globals() or len(df) == 0:
            print("No data available to run the analysis.")
            return
            
        # Plot the results
        plot_start_time = Time(times.iloc[0], format='unix').to_datetime().strftime('%H:%M:%S')
        plot_end_time = Time(times.iloc[-1], format='unix').to_datetime().strftime('%H:%M:%S')
        
        plt_times = Time(times, format='unix').to_datetime()
        plt.figure(figsize=(30, 18), dpi=300)
        plt.plot(plt_times, counts)
        
        plt.axhline(med, color='green', linestyle='-', linewidth=2,
                    label=f'Median: {med:.0f}')
        plt.axhline(med + sigma_widget.value * std, color='red', linestyle='-', linewidth=2,
                    label=f'{sigma_widget.value}σ Threshold: {med + sigma_widget.value * std:.0f}')
        
        # Annotate flares
        if 'flares' in globals() and flares:
            for i, flare in enumerate(flares, 1):
                t0, t1 = pd.to_datetime(flare['start_iso']), pd.to_datetime(flare['end_iso'])
                win = (plt_times >= t0) & (plt_times <= t1)
                if not win.any():
                    continue
                
                times_win = plt_times[win]
                counts_win = counts.iloc[win.nonzero()[0]]
                rel_idx = counts_win.values.argmax()
                t_max = times_win[rel_idx]
                y_max = counts_win.iloc[rel_idx]
                
                plt.scatter(t_max, y_max, color='magenta', s=60, zorder=5)
                plt.annotate(
                    f"Peak {i}\n{t_max:%H:%M:%S}",
                    xy=(t_max, y_max),
                    xytext=(-40, 0),
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    color='magenta',
                    fontsize=12,
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='magenta',
                        lw=1
                    )
                )
        
        plt.legend(fontsize=14)
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        plt.grid(True)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("X-Ray Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Get date values for title
        if hasattr(date_widget.value, 'strftime'):
            start_date_val = date_widget.value.strftime('%Y-%m-%d')
        else:
            start_date_val = date_widget.value
            
        if hasattr(end_date_widget.value, 'strftime'):
            end_date_val = end_date_widget.value.strftime('%Y-%m-%d')
        else:
            end_date_val = end_date_widget.value
            
        plt.title(
            f"SoLEXS X-ray Lightcurve\n\n"
            f"{start_date_val} {plot_start_time} to {end_date_val} {plot_end_time}",
            fontsize=16, fontweight='bold'
        )
        plt.show()

update_plot_button.on_click(update_light_curve)

display(update_plot_button)
display(plot_output)

# Run once to show initial plot
update_light_curve(None)




# Zoomed Flare Plot with Update Button
zoom_output = widgets.Output()
update_zoom_button = widgets.Button(
    description='Update',
    button_style='info',
    icon='refresh'
)

def update_zoomed_view(b):
    global df, times, counts, flares, zoom_window
    
    with zoom_output:
        clear_output()
        
        if 'df' not in globals() or len(df) == 0:
            print("No data available to run the analysis.")
            return
            
        if 'flares' not in globals() or not flares:
            print("No flares detected in the current data range.")
            return
        
        # Find the strongest flare
        peak_idx = np.nanargmax(df['COUNTS'])
        
        # Ensure indices are within bounds
        start = max(0, peak_idx - zoom_window*30)
        end = min(len(df), peak_idx + zoom_window*30)
        
        if end - start < 10:
            print("Insufficient data for zoomed view.")
            return
            
        time_subset = Time(df['TIME'].iloc[start:end], format='unix').to_datetime()
        
        plt.figure(figsize=(30, 18), dpi=300)
        plt.plot(time_subset, df['COUNTS'].iloc[start:end])
        plt.grid(True)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("X-Ray Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        date_str = df['DATE'].iloc[peak_idx].strftime('%Y-%m-%d')
        start_str = time_subset[0].strftime('%H:%M')
        end_str = time_subset[-1].strftime('%H:%M')
        
        plt.title(f"Zoomed-in Solar Flare\n\n{date_str} {start_str} to {end_str}", 
                  fontsize=16, fontweight='bold')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.show()

update_zoom_button.on_click(update_zoomed_view)

display(update_zoom_button)
display(zoom_output)

# Run once to show initial zoomed view
update_zoomed_view(None)

